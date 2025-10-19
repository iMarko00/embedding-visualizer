// server.js
import express from "express";

const app = express();
const port = 3000;

// Serve static files (our HTML, CSS, JS)
app.use(express.static("public"));

app.get("/data", (req, res) => {
  // Mock 10 2D vectors (pretend these are PCA/t-SNE results)
  const points = Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    x: Math.random() * 2 - 1,  // between -1 and 1
    y: Math.random() * 2 - 1,
    label: `Point ${i + 1}`
  }));
  res.json(points);
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});