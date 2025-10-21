import axios from "axios";
const MARQO_URL = "http://localhost:8882";
const INDEX_NAME = "earlybird_requirements";

export async function getEmbeddingsFromMarqo(requirements) {
  console.log("🧠 Letting Marqo embed requirements...");

  // Ensure index exists
  await axios.post(`${MARQO_URL}/indexes/${INDEX_NAME}`, {
    model: "sentence-transformers/all-MiniLM-L6-v2",
    type: "unstructured"
  })
  .catch(err => {
    if (err.response?.status === 409) {
      console.log("ℹ️ Index already exists.");
    } else {
      console.error("❌ Failed to create index:", err.response?.data || err.message);
      throw err;
    }
  });

  // Upload documents (Marqo embeds automatically)
  const docs = requirements.map((text, i) => ({
    _id: `req_${Date.now()}_${i}`,
    text
  }));

  console.log(`📤 Sending ${docs.length} requirements to Marqo...`);
  try {
  const res = await axios.post(`${MARQO_URL}/indexes/${INDEX_NAME}/documents`, {
    documents: docs,
    tensorFields: ["text"]
  });
  console.log("✅ Documents stored:", res.data);
    } catch (err) {
    console.error("❌ Failed sendind the reqs.:", err.response?.data || err.message);
    throw err;
    }

  console.log("✅ Documents sent successfully. Retrieving embeddings...");

  const vectors = [];

  for (const doc of docs) {
    try {
      const res = await axios.get(
        `${MARQO_URL}/indexes/${INDEX_NAME}/documents/${doc._id}?show_vectors=true`
      );
      // Inspect returned document to handle different Marqo versions/shapes
      console.log(`🔎 Document ${doc._id} returned:`, res.data);

      // possible shapes:
      // res.data._vectors = { text: [ ... ] }
      // res.data._vectors = { text: { values: [ ... ] } }
      // res.data._embedding (older)
      let vector = null;
      if (Array.isArray(res.data._vectors?.text)) {
        vector = res.data._vectors.text;
      } else if (Array.isArray(res.data._vectors?.text?.values)) {
        vector = res.data._vectors.text.values;
      } else if (Array.isArray(res.data._vectors?.text?.value)) {
        vector = res.data._vectors.text.value;
      } else if (Array.isArray(res.data._vectors?.text?.vector)) {
        vector = res.data._vectors.text.vector;
      } else if (Array.isArray(res.data._embedding)) {
        vector = res.data._embedding;
      } else if (res.data._vectors) {
        // try to find first array inside _vectors
        const first = Object.values(res.data._vectors).find(v => Array.isArray(v));
        if (first) vector = first;
      }

      if (vector && Array.isArray(vector)) {
        vectors.push(vector);
      } else {
        console.warn(`⚠️ No vector found for ${doc._id} (see logged document)`);
      }
    } catch (err) {
      console.error(`❌ Failed to fetch doc ${doc._id}:`, err.response?.data || err.message);
      throw err;
    }
  }

  if (vectors.length === 0) {
    throw new Error("No embeddings returned from Marqo!");
  }

  return vectors;
}