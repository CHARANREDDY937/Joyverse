// const express = require('express');
// const cors = require('cors');
// const bodyParser = require('body-parser');
// const mongoose = require('mongoose');
// const bcrypt = require('bcrypt');
// const axios = require('axios');
// require('dotenv').config();
// const themes = require("./themes");
// const fs = require('fs');
// const path = require('path');

// // Import the User model
// const User = require('./models/user');

// const app = express();

// // Environment-based FastAPI URL
// const isProduction = process.env.NODE_ENV === 'production';
// const FASTAPI_URL = isProduction
//   ? 'https://api-pmbi.onrender.com'
//   : 'http://127.0.0.1:8000';

// // Middlewares
// app.use(cors({
//   origin: ['https://joyverse.onrender.com', 'http://localhost:5173'],
//   methods: ['GET', 'POST', 'PUT', 'DELETE'],
//   allowedHeaders: ['Content-Type'],
// }));
// app.use(bodyParser.json());
// app.use(express.json());

// // MongoDB URI
// const MONGO_URI = process.env.MONGO_URI;

// // Connect to MongoDB
// mongoose.connect(MONGO_URI)
//   .then(() => console.log('Connected to MongoDB'))
//   .catch((err) => console.error('MongoDB connection error:', err));

// // Signup Route
// app.post('/api/signup', async (req, res) => {
//   const { username, password, confirmPassword, name } = req.body;
//   if (!username || !password || !confirmPassword || !name) {
//     return res.status(400).json({ message: 'All fields are required' });
//   }
//   if (password !== confirmPassword) {
//     return res.status(400).json({ message: 'Passwords do not match' });
//   }
//   try {
//     const existingUser = await User.findOne({ username });
//     if (existingUser) {
//       return res.status(400).json({ message: 'Username already taken' });
//     }
//     const hashedPassword = await bcrypt.hash(password, 10);
//     const newUser = new User({ username, password: hashedPassword, name });
//     await newUser.save();
//     console.log('New user saved:', newUser);
//     res.status(201).json({ message: 'User registered successfully', user: newUser });
//   } catch (err) {
//     console.error('Signup error:', err);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Login Route
// app.post('/api/login', async (req, res) => {
//   const { username, password } = req.body;
//   if (!username || !password) {
//     return res.status(400).json({ message: 'Username and password are required' });
//   }
//   try {
//     const user = await User.findOne({ username });
//     if (!user) {
//       return res.status(400).json({ message: 'User not found' });
//     }
//     const isMatch = await bcrypt.compare(password, user.password);
//     if (!isMatch) {
//       return res.status(400).json({ message: 'Incorrect password' });
//     }
//     res.status(200).json({ message: 'Login successful', user });
//   } catch (err) {
//     console.error('Login error:', err);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Emotion Summary Route
// app.post('/api/emotion-summary', async (req, res) => {
//   try {
//     console.log('Received /api/emotion-summary request:', req.body);
//     const jsonPath = path.join(__dirname, 'emotion_stats.json');
//     if (!fs.existsSync(jsonPath)) {
//       console.error('emotion_stats.json does not exist');
//       return res.status(400).json({ error: 'No emotion data found' });
//     }

//     let emotionData;
//     try {
//       const fileContent = fs.readFileSync(jsonPath, 'utf-8');
//       console.log('emotion_stats.json content:', fileContent);
//       emotionData = JSON.parse(fileContent);
//       // Validate emotion data
//       const requiredEmotions = ['Anger', 'Sadness', 'Happiness', 'Fear', 'Disgust', 'Surprise', 'Neutral'];
//       const isValidEmotion = requiredEmotions.every(
//         emotion => emotion in emotionData && typeof emotionData[emotion] === 'number' && !isNaN(emotionData[emotion])
//       );
//       if (!isValidEmotion || Object.keys(emotionData).length === 0) {
//         console.error('Invalid or empty emotion data:', emotionData);
//         return res.status(400).json({ error: 'Invalid or empty emotion data' });
//       }
//     } catch (parseError) {
//       console.error('Error parsing emotion_stats.json:', parseError.message);
//       return res.status(400).json({ error: 'Invalid emotion data format' });
//     }

//     const { username, game } = req.body;
//     if (!username || !game) {
//       console.error('Missing username or game in request:', { username, game });
//       return res.status(400).json({ error: 'Username and game are required' });
//     }

//     const user = await User.findOne({ username });
//     if (!user) {
//       console.error('User not found:', username);
//       return res.status(404).json({ error: 'User not found' });
//     }

//     // Format the summary entry
//     const summaryEntry = {
//       game,
//       emotion: {
//         Anger: emotionData.Anger ?? 0,
//         Sadness: emotionData.Sadness ?? 0,
//         Happiness: emotionData.Happiness ?? 0,
//         Fear: emotionData.Fear ?? 0,
//         Disgust: emotionData.Disgust ?? 0,
//         Surprise: emotionData.Surprise ?? 0,
//         Neutral: emotionData.Neutral ?? 0
//       },
//       timestamp: new Date() // Save as a Date object
//     };
//     console.log('Attempting to save summaryEntry:', summaryEntry);

//     // Push the new entry
//     user.emotionSummary = user.emotionSummary || [];
//     user.emotionSummary.push(summaryEntry);
//     await user.save();

//     console.log('Emotion data saved to user record:', user.emotionSummary);
//     fs.writeFileSync(jsonPath, JSON.stringify({}, null, 2));
//     console.log('emotion_stats.json cleared');

//     res.status(200).json({
//       message: 'Emotion summary updated successfully',
//       emotionSummary: user.emotionSummary
//     });
//   } catch (error) {
//     console.error('MongoDB save error:', error.message, error.stack);
//     res.status(500).json({ error: 'Failed to save emotion summary', details: error.message });
//   }
// });

// // Predict Emotion Route
// let emotionStats = {
//   Anger: 0,
//   Sadness: 0,
//   Happiness: 0,
//   Fear: 0,
//   Disgust: 0,
//   Surprise: 0,
//   Neutral: 0,
// };
// let totalPredictions = 0;

// const jsonPath = path.join(__dirname, 'emotion_stats.json');

// app.post('/api/predict-emotion', async (req, res) => {
//   try {
//     const { landmarks } = req.body;

//     // Validate landmarks
//     if (!Array.isArray(landmarks) || landmarks.length !== 468) {
//       console.error('Invalid landmark data: must be an array of 468 elements');
//       return res.status(400).json({ error: 'Invalid landmark data: must be an array of 468 elements' });
//     }
//     const isValidLandmarks = landmarks.every(
//       landmark => Array.isArray(landmark) && landmark.length === 3 && landmark.every(val => typeof val === 'number' && !isNaN(val))
//     );
//     if (!isValidLandmarks) {
//       console.error('Invalid landmark data: each landmark must be an array of 3 numbers');
//       return res.status(400).json({ error: 'Invalid landmark data: each landmark must be an array of 3 numbers' });
//     }

//     // Log landmarks for debugging
//     console.log('Sending landmarks to FastAPI, shape:', landmarks.length);

//     // Send request to FastAPI
//     const response = await axios.post(`${FASTAPI_URL}/predict`, { landmarks }, {
//       timeout: 10000 // 10-second timeout
//     });

//     if (response.status !== 200) {
//       throw new Error(`FastAPI error: ${response.status}`);
//     }

//     const { predicted_emotion } = response.data;
//     const themeUrl = themes[predicted_emotion] || themes['Neutral'];

//     if (emotionStats.hasOwnProperty(predicted_emotion)) {
//       emotionStats[predicted_emotion]++;
//       totalPredictions++;
//     } else {
//       console.warn('Unknown emotion received from FastAPI:', predicted_emotion);
//     }

//     const emotionPercentages = {};
//     for (const emotion in emotionStats) {
//       const percent = (emotionStats[emotion] / totalPredictions) * 100;
//       emotionPercentages[emotion] = parseFloat(percent.toFixed(2));
//     }

//     fs.writeFileSync(jsonPath, JSON.stringify(emotionPercentages, null, 2));
//     console.log('emotion_stats.json updated:', emotionPercentages);

//     res.status(200).json({
//       emotion: predicted_emotion,
//       theme: themeUrl,
//     });
//   } catch (error) {
//     console.error('Error calling FastAPI:', error.message, error.response ? error.response.data : '');
//     res.status(500).json({ error: 'Failed to get emotion prediction', details: error.response ? error.response.data : error.message });
//   }
// });

// // Fetch All Users
// app.get('/api/users', async (req, res) => {
//   try {
//     const users = await User.find();
//     res.json(users);
//   } catch (error) {
//     console.error('Error fetching users:', error);
//     res.status(500).json({ message: 'Error fetching user data' });
//   }
// });

// // View Progress Route
// app.get('/api/users/progress/:username', async (req, res) => {
//   const { username } = req.params;
//   console.log('Fetching progress for username:', username);
//   try {
//     const user = await User.findOne({ username });
//     if (!user) {
//       console.log('User not found:', username);
//       return res.status(404).json({ message: 'User not found' });
//     }

//     console.log('Found user:', user.username, 'Emotion summary:', user.emotionSummary);
//     res.json({
//       name: user.name,
//       username: user.username,
//       emotionSummary: user.emotionSummary || []
//     });
//   } catch (error) {
//     console.error('Error fetching user progress:', error);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Home Route (for testing)
// app.get('/api/node/home', (req, res) => {
//   res.status(200).json({ message: 'Node.js is running' });
// });

// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useChildContext } from '../context/ChildContext';
import './childlist.css';

// Determine API base URL based on environment
const isProduction = process.env.NODE_ENV === 'production';
const API_URL = isProduction
  ? 'https://backend-brmn.onrender.com'
  : 'http://localhost:5000';

const ChildList = () => {
  const navigate = useNavigate();
  const { setChildData, user } = useChildContext();
  const [users, setUsers] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');

  // Protect route
  useEffect(() => {
    if (!user || user.role !== 'therapist') {
      navigate('/therapist');
    }
  }, [user, navigate]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch(`${API_URL}/api/users`);
        if (!response.ok) {
          throw new Error('Failed to fetch user data');
        }
        const data = await response.json();
        setUsers(data);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.username.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleViewProgress = async (user) => {
    try {
      const response = await fetch(`${API_URL}/api/users/progress/${user.username}`);
      if (!response.ok) {
        throw new Error('Failed to fetch progress data');
      }
      const progressData = await response.json();

      setChildData({
        username: user.username,
        progressData,
        name: user.name
      });

      navigate(`/child-progress/${user.username}`);
    } catch (error) {
      console.error('Error fetching progress:', error);
      setError('Failed to load child progress. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loader"></div>
        <p>Loading children's data...</p>
      </div>
    );
  }

  return (
    <motion.div 
      className="child-list-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="dashboard-header">
        <h1>Children's Progress Dashboard</h1>
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search by name or username..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
      </div>

      {error ? (
        <motion.div 
          className="error-message"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {error}
        </motion.div>
      ) : (
        <div className="user-grid">
          {filteredUsers.map((user, index) => (
            <motion.div
              key={user._id}
              className="user-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="user-header">
                <div className="user-avatar">
                  <span>👤</span>
                </div>
                <div className="user-info">
                  <h3>{user.name}</h3>
                  <p>@{user.username}</p>
                </div>
              </div>

              {/* <div className="user-stats">
                <div className="stat">
                  <span>🎮</span>
                  <span>Games: {user.gamesPlayed || 0}</span>
                </div>
                <div className="stat">
                  <span>⭐</span>
                  <span>Score: {user.totalScore || 0}</span>
                </div>
              </div> */}

              <div className="button-group">
                <motion.button
                  className="view-progress-btn"
                  onClick={() => handleViewProgress(user)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <span>📈</span>
                  View Progress
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
};

export default ChildList;