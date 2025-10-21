import express from "express";
import { getEmbeddingsFromMarqo } from "../services/marqoService.js";
import { clusterRequirements } from "../services/clusteringService.js";
import { reduceTo2D } from "../services/pcaService.js";
import { nameClusters } from "../services/openaiService.js";

const router = express.Router();

router.post("/", async (req, res) => {
  const { requirements } = req.body;
  if (!requirements?.length) return res.status(400).json({ error: "No requirements provided" });

  try {
    console.log("📩 Received requirements:", requirements);

    // --- Phase 2 will replace this call ---
    const embeddings = await getEmbeddingsFromMarqo(requirements);

    const reduced = reduceTo2D(embeddings);
    const clusters = clusterRequirements(reduced, requirements.length);
    const result = await nameClusters(clusters, requirements, reduced);

    res.json(result);
  } catch (err) {
    console.error("❌ Embed route error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;