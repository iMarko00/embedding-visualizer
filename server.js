import express from "express";
import bodyParser from "body-parser";
import OpenAI from "openai";
import { PCA } from "ml-pca"; // for dimensionality reduction
import clustering from "density-clustering"; // for clustering

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
    const dbscan = new clustering.DBSCAN();
    // eps controls cluster radius, minPts = minimum points per cluster
    const clusters = dbscan.run(reduced, 0.3, 2);

    // Step 3 — Build cluster data
    const clusterMap = new Map();

    clusters.forEach((indices, clusterId) => {
      const clusterPoints = indices.map((i) => reduced[i]);
      const cx = clusterPoints.reduce((a, b) => a + b[0], 0) / clusterPoints.length;
      const cy = clusterPoints.reduce((a, b) => a + b[1], 0) / clusterPoints.length;
      const radius = Math.max(
        ...clusterPoints.map(([x, y]) => Math.hypot(x - cx, y - cy))
      );
      clusterMap.set(clusterId, { cx, cy, radius });
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
      clusters: Array.from(clusterMap.entries()).map(([id, { cx, cy, radius }]) => ({
        id,
        cx,
        cy,
        radius,
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