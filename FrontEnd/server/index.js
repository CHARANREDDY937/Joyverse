// server/index.js
const express = require("express");
const fs = require("fs");
const path = require("path");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json({ limit: "10mb" }));

// Ensure 'csv' folder exists
const csvDir = path.join(__dirname, "csv");
if (!fs.existsSync(csvDir)) fs.mkdirSync(csvDir);

app.post("/save-csv", (req, res) => {
  const { csv } = req.body;
  const filePath = path.join(csvDir, "landmark_data.csv");

  fs.writeFile(filePath, csv, (err) => {
    if (err) {
      console.error("Error writing CSV:", err);
      return res.status(500).send("Failed to save CSV.");
    }
    res.send("CSV saved to server/csv/landmark_data.csv");
  });
});

app.listen(PORT, () => console.log(`✅ Server running at http://localhost:${PORT}`));
