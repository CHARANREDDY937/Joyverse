const express= require('express')
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const cors = require('cors');
const User = require('./models/User');

const app = express();
app.use(express.json());
app.use(cors());

// MongoDB Connection
const mongoURI = "mongodb://localhost:27017/mydatabase" // Replace with your MongoDB URI
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('MongoDB connected'))
  .catch(err => console.error('MongoDB connection error:', err));

// Register User Route
app.post('/register', async (req, res) => {
  try {
    const { name, password } = req.body;
    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ name, password: hashedPassword });
    await user.save();
    res.status(201).json({ message: 'User registered successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Server error' });
  }
});

const PORT = 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
