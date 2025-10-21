import express from "express";
import bodyParser from "body-parser";
import embedRouter from "./routes/embed.js";

const app = express();
app.use(bodyParser.json());
app.use(express.static("public"));
app.use("/embed", embedRouter);

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`🚀 Server running at http://localhost:${port}`));