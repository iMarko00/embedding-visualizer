import express from "express";
import bodyParser from "body-parser";
import OpenAI from "openai";
import { PCA } from "ml-pca";

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

    // Only run PCA if we have at least 2 unique embeddings
    if (embeddings.length < 2) {
      console.warn("⚠️ Not enough data for PCA, returning flat points.");
      return res.json(
        requirements.map((req, i) => ({
          text: req,
          x: 0,
          y: i,
        }))
      );
    }

  // Defensive copy so ml-pca can't mutate
  const pca = new PCA(embeddings);
  const nComp = Math.min(2, embeddings.length);
  const reduced = pca.predict(embeddings, { nComponents: nComp }).to2DArray();

  const points = requirements.map((req, i) => ({
    text: req,
    x: reduced[i][0],
    y: reduced[i][1],
  }));

  res.json(points);

  } catch (err) {
    console.error("❌ Error during embedding:", err);
    res.status(500).json({ error: "Embedding failed", details: err.message });
  }
});

app.listen(port, () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});