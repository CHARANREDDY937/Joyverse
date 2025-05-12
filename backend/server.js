// const express = require('express');
// const cors = require('cors');
// const bodyParser = require('body-parser');
// const mongoose = require('mongoose');
// const bcrypt = require('bcrypt');
// require('dotenv').config();

// const app = express();
// app.use(cors());
// app.use(bodyParser.json()); // Middleware for parsing JSON requests

// // Connect to MongoDB
// const MONGO_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/childlogin';

// mongoose.connect(MONGO_URI, {
//   useNewUrlParser: true,
//   useUnifiedTopology: true,
// })
//   .then(() => console.log(`✅ Connected to MongoDB at ${MONGO_URI}`))
//   .catch((err) => console.error('❌ MongoDB connection error:', err));

// // Schema and Model for User
// const userSchema = new mongoose.Schema({
//   username: { type: String, required: true, unique: true },
//   password: { type: String, required: true },
//   name: { type: String, required: true },
// });

// const User = mongoose.model('User', userSchema);

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
//     console.log('✅ New user saved to MongoDB:', newUser);

//     res.status(201).json({ message: 'User registered successfully', user: newUser });
//   } catch (err) {
//     console.error('❌ Signup error:', err);
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
//     console.error('❌ Login error:', err);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Endpoint to fetch all users (renamed from children)
// app.get('/api/users', async (req, res) => {
//   try {
//     const users = await User.find(); // Fetch all users from MongoDB
//     res.json(users);
//   } catch (error) {
//     console.error('❌ Error fetching users:', error);
//     res.status(500).json({ message: 'Error fetching user data' });
//   }
// });
// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));



// const express = require('express');
// const cors = require('cors');
// const bodyParser = require('body-parser');
// const mongoose = require('mongoose');
// const bcrypt = require('bcrypt');
// require('dotenv').config();

// const app = express();
// app.use(cors());
// app.use(bodyParser.json()); // Middleware for parsing JSON requests

// // ✅ Correct MongoDB URI (password special character encoded)
// const MONGO_URI = "mongodb+srv://sushmasreebandi92:Ramuindu%40123@cluster0.exp0qcv.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";

// // ✅ Connect to MongoDB (modern Mongoose doesn't need options)
// mongoose.connect(MONGO_URI)
//   .then(() => console.log(`✅ Connected to MongoDB at ${MONGO_URI}`))
//   .catch((err) => console.error('❌ MongoDB connection error:', err));

// // Schema and Model for User
// const userSchema = new mongoose.Schema({
//   username: { type: String, required: true, unique: true },
//   password: { type: String, required: true },
//   name: { type: String, required: true },
// });

// const User = mongoose.model('User', userSchema);

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
//     console.log('✅ New user saved to MongoDB:', newUser);

//     res.status(201).json({ message: 'User registered successfully', user: newUser });
//   } catch (err) {
//     console.error('❌ Signup error:', err);
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
//     console.error('❌ Login error:', err);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Endpoint to fetch all users
// app.get('/api/users', async (req, res) => {
//   try {
//     const users = await User.find(); // Fetch all users from MongoDB
//     res.json(users);
//   } catch (error) {
//     console.error('❌ Error fetching users:', error);
//     res.status(500).json({ message: 'Error fetching user data' });
//   }
// });

// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));


// const express = require('express');
// const cors = require('cors');
// const bodyParser = require('body-parser');
// const mongoose = require('mongoose');
// const bcrypt = require('bcrypt');
// const axios = require('axios'); // Ensure axios is installed using npm i axios
// require('dotenv').config(); // Load environment variables from .env
// const themes = require("./themes");
// const fs = require('fs');
// const path = require('path');

// const app = express();

// // Middlewares
// app.use(cors());
// app.use(bodyParser.json()); // Parse incoming JSON
// app.use(express.json());


// // ✅ Use MongoDB URI from environment variables
// const MONGO_URI = process.env.MONGO_URI;

// // Connect to MongoDB
// mongoose.connect(MONGO_URI)
//   .then(() => console.log('✅ Connected to MongoDB'))
//   .catch((err) => console.error('❌ MongoDB connection error:', err));

// // Define User Schema and Model
// const userSchema = new mongoose.Schema({
//   username: { type: String, required: true, unique: true },
//   password: { type: String, required: true },
//   name: { type: String, required: true },
//   emotionSummary: [
//     {
//       game: { type: String, required: true },
//       emotion: {
//         Sadness: { type: Number, required: true },
//         Happiness: { type: Number, required: true },
//         Fear: { type: Number, required: true },
//         Disgust: { type: Number, required: true },
//         Surprise: { type: Number, required: true },
//         Neutral: { type: Number, required: true }
//       },
//       timestamp: { type: Date, required: true }
//     }
//   ]
// });

// const User = mongoose.model('User', userSchema);

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
//     console.log('✅ New user saved:', newUser);

//     res.status(201).json({ message: 'User registered successfully', user: newUser });
//   } catch (err) {
//     console.error('❌ Signup error:', err);
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
//     console.error('❌ Login error:', err);
//     res.status(500).json({ message: 'Internal server error' });
//   }
// });

// // Emotion Summary Route
// app.post('/api/emotion-summary', async (req, res) => {
//   try {
//     const jsonPath = path.join(__dirname, 'emotion_stats.json');

//     // Check if JSON file exists
//     if (!fs.existsSync(jsonPath)) {
//       return res.status(400).json({ error: "No emotion data found" });
//     }

//     // Read the emotion data from JSON file
//     const emotionData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

//     const { username, game } = req.body;

//     // Find user in MongoDB
//     const user = await User.findOne({ username });
//     if (!user) {
//       return res.status(400).json({ error: 'User not found' });
//     }

//     // Append emotion data to the user's emotion summary
//     user.emotionSummary.push({
//       game,
//       emotion: emotionData,
//       timestamp: new Date().toISOString()
//     });

//     // Save the updated user
//     await user.save();
//     console.log("✅ Emotion data saved to user record");

//     // Clear the JSON file (overwrite with an empty object)
//     fs.writeFileSync(jsonPath, JSON.stringify({}, null, 2));
//     console.log("📂 Emotion data JSON file cleared");

//     // Send response back to the frontend
//     res.status(200).json({
//       message: 'Emotion summary updated successfully',
//       emotionSummary: user.emotionSummary
//     });
//   } catch (error) {
//     console.error("❌ Error in emotion-summary route:", error.message);
//     res.status(500).json({ error: "Failed to update emotion summary" });
//   }
// });

// // Predict Emotion Route
// let emotionStats = {
//   Sadness: 0,
//   Happiness: 0,
//   Fear: 0,
//   Disgust: 0,
//   Surprise: 0,
//   Neutral: 0,
// };
// let totalPredictions = 0;

// // JSON output path
// const jsonPath = path.join(__dirname, 'emotion_stats.json');

// app.post('/api/predict-emotion', async (req, res) => {
//   try {
//     const landmarks = req.body.landmarks;

//     if (!Array.isArray(landmarks) || landmarks.length !== 468) {
//       return res.status(400).json({ error: "Invalid landmark data" });
//     }

//     const fastApiUrl = "http://127.0.0.1:8000/predict";

//     const response = await axios.post(fastApiUrl, { landmarks });

//     if (response.status !== 200) {
//       throw new Error(`FastAPI error: ${response.status}`);
//     }

//     const { predicted_emotion } = response.data;
//     const themeUrl = themes[predicted_emotion] || themes["Neutral"];

//     // Update stats
//     if (emotionStats.hasOwnProperty(predicted_emotion)) {
//       emotionStats[predicted_emotion]++;
//       totalPredictions++;
//     }

//     // Calculate percentages
//     const emotionPercentages = {};
//     for (const emotion in emotionStats) {
//       const percent = (emotionStats[emotion] / totalPredictions) * 100;
//       emotionPercentages[emotion] = parseFloat(percent.toFixed(2));
//     }

//     // Write to JSON file
//     fs.writeFileSync(jsonPath, JSON.stringify(emotionPercentages, null, 2));
//     // console.log("📊 Emotion stats saved to JSON");

//     // Send response to frontend
//     res.status(200).json({
//       emotion: predicted_emotion,
//       theme: themeUrl,
//     });

//   } catch (error) {
//     console.error("❌ Error calling FastAPI:", error.message);
//     res.status(500).json({ error: "Failed to get emotion prediction" });
//   }
// });

// // Users Route (Fetch All Users)
// app.get('/api/users', async (req, res) => {
//   try {
//     const users = await User.find(); // Get all users
//     res.json(users);
//   } catch (error) {
//     console.error('❌ Error fetching users:', error);
//     res.status(500).json({ message: 'Error fetching user data' });
//   }
// });

// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
// server.js
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
app.use(bodyParser.json()); // Parse incoming JSON
app.use(express.json());

// ✅ Use MongoDB URI from environment variables
const MONGO_URI = process.env.MONGO_URI;

// Connect to MongoDB
mongoose.connect(MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB'))
  .catch((err) => console.error('❌ MongoDB connection error:', err));

// Signup Route
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

// Login Route
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

// Emotion Summary Route
app.post('/api/emotion-summary', async (req, res) => {
  try {
    const jsonPath = path.join(__dirname, 'emotion_stats.json');

    // Check if JSON file exists
    if (!fs.existsSync(jsonPath)) {
      return res.status(400).json({ error: "No emotion data found" });
    }

    // Read the emotion data from JSON file
    const emotionData = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));

    const { username, game } = req.body;

    // Find user in MongoDB
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(400).json({ error: 'User not found' });
    }

    // Append emotion data to the user's emotion summary
    user.emotionSummary.push({
      game,
      emotion: emotionData,
      timestamp: new Date().toISOString()
    });

    // Save the updated user
    await user.save();
    console.log("✅ Emotion data saved to user record");

    // Clear the JSON file (overwrite with an empty object)
    fs.writeFileSync(jsonPath, JSON.stringify({}, null, 2));
    console.log("📂 Emotion data JSON file cleared");

    // Send response back to the frontend
    res.status(200).json({
      message: 'Emotion summary updated successfully',
      emotionSummary: user.emotionSummary
    });
  } catch (error) {
    console.error("❌ Error in emotion-summary route:", error.message);
    res.status(500).json({ error: "Failed to update emotion summary" });
  }
});

// Predict Emotion Route
let emotionStats = {
  Sadness: 0,
  Happiness: 0,
  Fear: 0,
  Disgust: 0,
  Surprise: 0,
  Neutral: 0,
};
let totalPredictions = 0;

// JSON output path
const jsonPath = path.join(__dirname, 'emotion_stats.json');

app.post('/api/predict-emotion', async (req, res) => {
  try {
    const landmarks = req.body.landmarks;

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

    // Update stats
    if (emotionStats.hasOwnProperty(predicted_emotion)) {
      emotionStats[predicted_emotion]++;
      totalPredictions++;
    }

    // Calculate percentages
    const emotionPercentages = {};
    for (const emotion in emotionStats) {
      const percent = (emotionStats[emotion] / totalPredictions) * 100;
      emotionPercentages[emotion] = parseFloat(percent.toFixed(2));
    }

    // Write to JSON file
    fs.writeFileSync(jsonPath, JSON.stringify(emotionPercentages, null, 2));

    // Send response to frontend
    res.status(200).json({
      emotion: predicted_emotion,
      theme: themeUrl,
    });

  } catch (error) {
    console.error("❌ Error calling FastAPI:", error.message);
    res.status(500).json({ error: "Failed to get emotion prediction" });
  }
});

// Users Route (Fetch All Users)
app.get('/api/users', async (req, res) => {
  try {
    const users = await User.find(); // Get all users
    res.json(users);
  } catch (error) {
    console.error('❌ Error fetching users:', error);
    res.status(500).json({ message: 'Error fetching user data' });
  }
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
