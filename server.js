import express from "express";
import bodyParser from "body-parser";
import OpenAI from "openai";
import { PCA } from "ml-pca"; // for dimensionality reduction
import { DBSCAN } from "density-clustering"; // for clustering

const app = express();
const port = 3000;

app.use(express.static("public"));
app.use(bodyParser.json());

console.log("🔑 OpenAI API key loaded?", !!process.env.OPENAI_API_KEY);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.post("/embed", async (req, res) => {
  const { requirements } = req.body;
  console.log("📩 Received requirements:", requirements);

  if (!requirements || requirements.length === 0) {
    console.warn("⚠️ No requirements provided!");
    return res.status(400).json({ error: "No requirements provided" });
  }

  try {
    console.log("🧠 Requesting embeddings from OpenAI...");
    const response = await client.embeddings.create({
      model: "text-embedding-3-small",
      input: requirements,
    });
    console.log("✅ Embeddings received!");
    console.log("🔍 Embeddings:", response.data);

    const embeddings = response.data.map((item) => item.embedding);

    // Step 1 — PCA reduction (1536D → 2D)
    const pca = new PCA(embeddings);
    const reduced = pca.predict(embeddings, { nComponents: 2 }).to2DArray();

    // Step 2 — Run DBSCAN on the reduced 2D points
    const dbscan = new DBSCAN();
    // eps controls cluster radius, minPts = minimum points per cluster
    const clusters = dbscan.run(reduced, 0.3, 2);

    // Step 3 — Build cluster data and generate names
    const clusterMap = new Map();

    // Generate cluster names using OpenAI
    console.log("🏷️ Generating cluster names with OpenAI...");
    const clusterNames = await Promise.all(
      clusters.map(async (indices, clusterId) => {
        const clusterRequirements = indices.map(i => requirements[i]);
        console.log(`📝 Cluster ${clusterId} requirements:`, clusterRequirements);
        
        try {
          const completion = await client.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
              {
                role: "system",
                content: "You are an expert at analyzing software requirements and grouping them into meaningful categories. Given a list of related requirements, provide a concise, descriptive name (2-4 words) that captures the common theme or functionality. Examples: 'User Authentication', 'Payment Processing', 'Data Management', 'UI Components', 'Security Features'."
              },
              {
                role: "user",
                content: `Analyze these requirements and suggest a concise name for this cluster:\n\n${clusterRequirements.map((req, i) => `${i + 1}. ${req}`).join('\n')}\n\nProvide only the cluster name, nothing else.`
              }
            ],
            max_tokens: 20,
            temperature: 0.3
          });
          
          const name = completion.choices[0].message.content.trim().replace(/['"]/g, '');
          console.log(`✅ Cluster ${clusterId} named: "${name}"`);
          return name;
        } catch (error) {
          console.warn(`⚠️ Failed to generate name for cluster ${clusterId}:`, error.message);
          return `Cluster ${clusterId}`;
        }
      })
    );

    clusters.forEach((indices, clusterId) => {
      const clusterPoints = indices.map((i) => reduced[i]);
      const cx = clusterPoints.reduce((a, b) => a + b[0], 0) / clusterPoints.length;
      const cy = clusterPoints.reduce((a, b) => a + b[1], 0) / clusterPoints.length;
      const radius = Math.max(
        ...clusterPoints.map(([x, y]) => Math.hypot(x - cx, y - cy))
      );
      clusterMap.set(clusterId, { 
        cx, 
        cy, 
        radius, 
        name: clusterNames[clusterId] || `Cluster ${clusterId}` 
      });
    });

    // Step 4 — Combine reduced points with their cluster assignments
    const labeledPoints = reduced.map(([x, y], i) => {
      let clusterId = -1;
      clusters.forEach((indices, id) => {
        if (indices.includes(i)) clusterId = id;
      });
      return { text: requirements[i], x, y, cluster: clusterId };
    });

    res.json({
      points: labeledPoints,
      clusters: Array.from(clusterMap.entries()).map(([id, { cx, cy, radius, name }]) => ({
        id,
        cx,
        cy,
        radius,
        name,
      })),
    });
  } catch (err) {
    console.error("❌ Error during embedding:", err);
    res.status(500).json({ error: "Embedding failed", details: err.message });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});