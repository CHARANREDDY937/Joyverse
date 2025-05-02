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
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const axios = require('axios'); // Ensure axios is installed using npm i axios
require('dotenv').config({ path: __dirname + '/.env' });
const themes = require("./themes");

const app = express();

// Middlewares
app.use(cors());
app.use(bodyParser.json()); // Parse incoming JSON

// ✅ Use MongoDB URI from environment variables
const MONGO_URI = process.env.MONGO_URI;

// Connect to MongoDB
mongoose.connect(MONGO_URI)
  .then(() => console.log(`✅ Connected to MongoDB`))
  .catch((err) => console.error('❌ MongoDB connection error:', err));

// Define User Schema and Model
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String, required: true },
});

const User = mongoose.model('User', userSchema);

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

// app.post('/api/predict-emotion', async (req, res) => {
//   try {
//     const landmarks = req.body.landmarks;
//     if (!Array.isArray(landmarks) || landmarks.length !== 468) {
//       return res.status(400).json({ error: "Invalid landmark data" });
//     }

//     const fastApiUrl = "http://127.0.0.1:8000/predict";

//     const response = await fetch(fastApiUrl, {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({ landmarks }),
//     });

//     if (!response.ok) {
//       throw new Error(`FastAPI error: ${response.status}`);
//     }

//     const prediction = await response.json();

//     // ✅ Log the prediction received from FastAPI
//     console.log("🎯 Prediction from FastAPI:", prediction);

//     // Send the prediction to frontend
//     res.status(200).json(prediction);
//   } catch (error) {
//     console.error("❌ Error calling FastAPI:", error.message);
//     res.status(500).json({ error: "Failed to get emotion prediction" });
//   }
// });
app.post('/api/predict-emotion', async (req, res) => {
  try {
    const landmarks = req.body.landmarks;
    if (!Array.isArray(landmarks) || landmarks.length !== 468) {
      return res.status(400).json({ error: "Invalid landmark data" });
    }

    const fastApiUrl = "http://127.0.0.1:8000/predict";

    const response = await fetch(fastApiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ landmarks }),
    });

    if (!response.ok) {
      throw new Error(`FastAPI error: ${response.status}`);
    }

    const { predicted_emotion } = await response.json();
    const themeUrl = themes[predicted_emotion] || themes["Neutral"];

    console.log("🎯 Emotion:", predicted_emotion);
    console.log("🎨 Theme URL:", themeUrl);

    // Return both emotion and theme image URL to frontend
    res.status(200).json({ emotion: predicted_emotion, theme: themeUrl });
  } catch (error) {
    console.error("❌ Error calling FastAPI:", error.message);
    res.status(500).json({ error: "Failed to get emotion prediction" });
  }
});


// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));
