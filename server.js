import express from "express";
import bodyParser from "body-parser";
import OpenAI from "openai";
import { PCA } from "ml-pca"; // for dimensionality reduction
import { DBSCAN } from "density-clustering"; // for clustering
import { kmeans } from "ml-kmeans"; // fallback clustering

const app = express();
const port = 3000;

app.use(express.static("public"));
app.use(bodyParser.json());

console.log("🔑 OpenAI API key loaded?", !!process.env.OPENAI_API_KEY);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Helper function to process clusters and generate names
async function processClusters(clusters, requirements, reduced, client) {
  const clusterMap = new Map();

  // Generate cluster names using OpenAI
  console.log("🏷️ Generating cluster names with OpenAI...");
  const clusterNames = await Promise.all(
    Array.from(clusters.entries()).map(async ([clusterId, indices]) => {
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
        return { clusterId, name };
      } catch (error) {
        console.warn(`⚠️ Failed to generate name for cluster ${clusterId}:`, error.message);
        return { clusterId, name: `Cluster ${clusterId}` };
      }
    })
  );

  // Create a name lookup map
  const nameMap = new Map();
  clusterNames.forEach(({ clusterId, name }) => {
    nameMap.set(clusterId, name);
  });

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
      name: nameMap.get(clusterId) || `Cluster ${clusterId}` 
    });
  });

  // Combine reduced points with their cluster assignments
  const labeledPoints = reduced.map(([x, y], i) => {
    let clusterId = -1;
    clusters.forEach((indices, id) => {
      if (indices.includes(i)) clusterId = id;
    });
    return { text: requirements[i], x, y, cluster: clusterId };
  });

  return {
    points: labeledPoints,
    clusters: Array.from(clusterMap.entries()).map(([id, { cx, cy, radius, name }]) => ({
      id,
      cx,
      cy,
      radius,
      name,
    })),
  };
}

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
    
    // Adaptive clustering parameters based on number of requirements
    const numRequirements = requirements.length;
    let eps, minPts;
    
    if (numRequirements <= 5) {
      eps = 0.2;
      minPts = 2;
    } else if (numRequirements <= 15) {
      eps = 0.2;
      minPts = 3;
    } else if (numRequirements <= 30) {
      eps = 0.15;
      minPts = 5;
    } else {
      // For large datasets, skip DBSCAN and go straight to K-means
      console.log("🔄 Large dataset detected, using K-means directly...");
      console.log(`📊 Debug: numRequirements = ${numRequirements}`);
      console.log(`📊 Debug: reduced array length = ${reduced.length}`);
      console.log(`📊 Debug: reduced[0] =`, reduced[0]);
      
      const k = Math.min(Math.max(6, Math.floor(numRequirements / 4)), Math.floor(numRequirements / 2)); // 6 to n/2 clusters
      console.log(`🔧 Using K-means with k=${k} clusters for ${numRequirements} points`);
      console.log(`📊 Debug: k value = ${k} (type: ${typeof k})`);
      console.log(`📊 Debug: reduced array type = ${Array.isArray(reduced) ? 'array' : typeof reduced}`);
      
      const kmeansResult = kmeans(reduced, k);
      console.log(`📊 Debug: kmeansResult type =`, typeof kmeansResult);
      console.log(`📊 Debug: kmeansResult =`, kmeansResult);
      
      // Convert K-means result to Map format
      const clusters = new Map();
      const clusterAssignments = kmeansResult.clusters; // K-means returns { clusters: [...] }
      for (let i = 0; i < k; i++) {
        const clusterIndices = [];
        clusterAssignments.forEach((clusterId, index) => {
          if (clusterId === i) {
            clusterIndices.push(index);
          }
        });
        if (clusterIndices.length > 0) {
          clusters.set(i, clusterIndices);
        }
      }
      
      console.log(`📊 K-means results: ${clusters.size} clusters found`);
      clusters.forEach((indices, clusterId) => {
        console.log(`   Cluster ${clusterId}: ${indices.length} requirements`);
      });
      
      // Skip the rest of the DBSCAN logic and go directly to cluster naming
      const result = await processClusters(clusters, requirements, reduced, client);
      return res.json(result);
    }
    
    console.log(`🔧 Using DBSCAN parameters: eps=${eps}, minPts=${minPts} for ${numRequirements} requirements`);
    const dbscanResult = dbscan.run(reduced, eps, minPts);
    
    // Debug clustering results
    console.log(`📊 DBSCAN result type:`, typeof dbscanResult);
    console.log(`📊 DBSCAN result:`, dbscanResult);
    
    // Convert DBSCAN result to Map format
    const clusters = new Map();
    if (Array.isArray(dbscanResult)) {
      dbscanResult.forEach((indices, clusterId) => {
        if (indices && indices.length > 0) {
          clusters.set(clusterId, indices);
        }
      });
    }
    
    console.log(`📊 Clustering results: ${clusters.size} clusters found`);
    clusters.forEach((indices, clusterId) => {
      console.log(`   Cluster ${clusterId}: ${indices.length} requirements`);
    });
    
    // Fallback: If DBSCAN produces only one cluster, use K-means instead
    if (clusters.size <= 1 && numRequirements > 5) {
      console.log("🔄 DBSCAN produced only one cluster, falling back to K-means...");
      console.log(`📊 Debug: numRequirements = ${numRequirements}`);
      console.log(`📊 Debug: reduced array length = ${reduced.length}`);
      
      const k = Math.min(Math.max(6, Math.floor(numRequirements / 4)), Math.floor(numRequirements / 2)); // 6 to n/2 clusters
      console.log(`🔧 Using K-means with k=${k} clusters for ${numRequirements} points`);
      console.log(`📊 Debug: k value = ${k} (type: ${typeof k})`);
      
      const kmeansResult = kmeans(reduced, k);
      
      // Convert K-means result to DBSCAN-like format
      clusters.clear();
      const clusterAssignments = kmeansResult.clusters; // K-means returns { clusters: [...] }
      for (let i = 0; i < k; i++) {
        const clusterIndices = [];
        clusterAssignments.forEach((clusterId, index) => {
          if (clusterId === i) {
            clusterIndices.push(index);
          }
        });
        if (clusterIndices.length > 0) {
          clusters.set(i, clusterIndices);
        }
      }
      
      console.log(`📊 K-means results: ${clusters.size} clusters found`);
      clusters.forEach((indices, clusterId) => {
        console.log(`   Cluster ${clusterId}: ${indices.length} requirements`);
      });
    }

    // Process clusters and generate names
    const result = await processClusters(clusters, requirements, reduced, client);
    res.json(result);
  } catch (err) {
    console.error("❌ Error during embedding:", err);
    res.status(500).json({ error: "Embedding failed", details: err.message });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});