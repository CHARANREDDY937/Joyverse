const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const axios = require('axios');
require('dotenv').config(); // Load environment variables from .env
const themes = require("./themes");
const fs = require('fs');
const path = require('path');

// Import the User model
const User = require('./models/user');

const app = express();

// Middlewares
app.use(cors());
app.use(bodyParser.json());
app.use(express.json());

// MongoDB URI
const MONGO_URI = process.env.MONGO_URI;

// Connect to MongoDB
mongoose.connect(MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch((err) => console.error('❌ MongoDB connection error:', err));

/* ------------------------ Signup Route ------------------------ */
app.post('/api/signup', async (req, res) => {
  const { username, password, confirmPassword, name } = req.body;

  if (!username || !password || !confirmPassword || !name) {
    return res.status(400).json({ message: 'All fields are required' });
  }

  if (password !== confirmPassword) {
    return res.status(400).json({ message: 'Passwords do not match' });
  }

  try {
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return res.status(400).json({ message: 'Username already taken' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({ username, password: hashedPassword, name });

    await newUser.save();
    console.log('✅ New user saved:', newUser);

    res.status(201).json({ message: 'User registered successfully', user: newUser });
  } catch (err) {
    console.error('❌ Signup error:', err);
    res.status(500).json({ message: 'Internal server error' });
  }
});

/* ------------------------ Login Route ------------------------ */
app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;

  if (!username || !password) {
    return res.status(400).json({ message: 'Username and password are required' });
  }

  try {
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(400).json({ message: 'User not found' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: 'Incorrect password' });
    }

    res.status(200).json({ message: 'Login successful', user });
  } catch (err) {
    console.error('❌ Login error:', err);
    res.status(500).json({ message: 'Internal server error' });
  }
});

/* ---------------- Emotion Summary Route ---------------- */
app.post('/api/emotion-summary', async (req, res) => {
  try {
    const jsonPath = path.join(__dirname, 'emotion_stats.json');

    if (!fs.existsSync(jsonPath)) {
      return res.status(400).json({ error: "No emotion data found" });
    }

    const emotionData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    const { username, game } = req.body;

    const user = await User.findOne({ username });
    if (!user) {
      return res.status(400).json({ error: 'User not found' });
    }

    user.emotionSummary.push({
      game,
      emotion: emotionData,
      timestamp: new Date().toISOString()
    });

    await user.save();
    console.log("✅ Emotion data saved to user record");

    fs.writeFileSync(jsonPath, JSON.stringify({}, null, 2));
    console.log("📂 Emotion data JSON file cleared");

    res.status(200).json({
      message: 'Emotion summary updated successfully',
      emotionSummary: user.emotionSummary
    });
  } catch (error) {
    console.error("❌ Error in emotion-summary route:", error.message);
    res.status(500).json({ error: "Failed to update emotion summary" });
  }
});

/* ---------------- Predict Emotion Route ---------------- */
let emotionStats = {
  Sadness: 0,
  Happiness: 0,
  Fear: 0,
  Disgust: 0,
  Surprise: 0,
  Neutral: 0,
};
let totalPredictions = 0;

const jsonPath = path.join(__dirname, 'emotion_stats.json');

app.post('/api/predict-emotion', async (req, res) => {
  try {
    const { landmarks } = req.body;

    if (!Array.isArray(landmarks) || landmarks.length !== 468) {
      return res.status(400).json({ error: "Invalid landmark data" });
    }

    const fastApiUrl = "http://127.0.0.1:8000/predict";
    const response = await axios.post(fastApiUrl, { landmarks });

    if (response.status !== 200) {
      throw new Error(`FastAPI error: ${response.status}`);
    }

    const { predicted_emotion } = response.data;
    const themeUrl = themes[predicted_emotion] || themes["Neutral"];

    if (emotionStats.hasOwnProperty(predicted_emotion)) {
      emotionStats[predicted_emotion]++;
      totalPredictions++;
    }

    const emotionPercentages = {};
    for (const emotion in emotionStats) {
      const percent = (emotionStats[emotion] / totalPredictions) * 100;
      emotionPercentages[emotion] = parseFloat(percent.toFixed(2));
    }

    fs.writeFileSync(jsonPath, JSON.stringify(emotionPercentages, null, 2));

    res.status(200).json({
      emotion: predicted_emotion,
      theme: themeUrl,
    });

  } catch (error) {
    console.error("❌ Error calling FastAPI:", error.message);
    res.status(500).json({ error: "Failed to get emotion prediction" });
  }
});

/* ---------------- Fetch All Users ---------------- */
app.get('/api/users', async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    console.error('❌ Error fetching users:', error);
    res.status(500).json({ message: 'Error fetching user data' });
  }
});

/* ---------------- View Progress Route ---------------- */
app.get('/api/users/progress/:username', async (req, res) => {
  const { username } = req.params;
  console.log('Fetching progress for username:', username); // Add logging
  try {
    const user = await User.findOne({ username });
    if (!user) {
      console.log('User not found:', username); // Add logging
      return res.status(404).json({ message: 'User not found' });
    }

    console.log('Found user:', user.username); // Add logging
    res.json({
      name: user.name,
      username: user.username,
      emotionSummary: user.emotionSummary || []
    });
  } catch (error) {
    console.error('❌ Error fetching user progress:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

/* ---------------- Start Server ---------------- */
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
