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



const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(bodyParser.json()); // Middleware for parsing JSON requests

// Connect to MongoDB Atlas
const MONGO_URI = process.env.MONGODB_URI || 'mongodb+srv://bannuru_charan_reddy:RB1817BA@cluster1.q9nhb.mongodb.net/childlogin?retryWrites=true&w=majority&appName=Cluster1';

mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log(`✅ Connected to MongoDB Atlas at ${MONGO_URI}`))
  .catch((err) => console.error('❌ MongoDB connection error:', err));

// Schema and Model for User
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
    console.log('✅ New user saved to MongoDB:', newUser);

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

// Endpoint to fetch all users
app.get('/api/users', async (req, res) => {
  try {
    const users = await User.find(); // Fetch all users from MongoDB
    res.json(users);
  } catch (error) {
    console.error('❌ Error fetching users:', error);
    res.status(500).json({ message: 'Error fetching user data' });
  }
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Server running at http://localhost:${PORT}`));