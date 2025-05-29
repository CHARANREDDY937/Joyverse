const express = require('express');
const router = express.Router();
const multer = require('multer');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken'); // Added for token generation
const Therapist = require('../models/Therapist');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: function (req, file, cb) {
    cb(null, `${Date.now()}-${file.originalname}`); // Added backticks for string interpolation
  }
});

const upload = multer({ storage: storage });

// Login route
router.post('/login', upload.none(), async (req, res) => {
  try {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.status(400).json({ message: 'Username and password are required' });
    }

    const therapist = await Therapist.findOne({ username });
    
    if (!therapist) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    const isMatch = await bcrypt.compare(password, therapist.password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Check verification status
    if (!therapist.isVerified) {
      return res.status(403).json({ 
        message: 'Your account is pending verification. Please wait for admin approval.',
        isVerified: false
      });
    }

    // Generate JWT token
    const token = jwt.sign(
      { id: therapist._id, role: 'therapist' },
      process.env.JWT_SECRET || 'your-secret-key', // Use environment variable for secret
      { expiresIn: '24h' }
    );

    res.json({ 
      message: 'Login successful',
      token, // Replaced placeholder with actual token
      isVerified: true,
      therapist: {
        username: therapist.username,
        fullName: therapist.fullName,
        email: therapist.email
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error' });
  }
});

// Register route
router.post('/register', upload.single('certificate'), async (req, res) => {
  try {
    const { username, password, fullName, email, degree, specialization } = req.body;
    
    // Validate required fields
    if (!username || !password || !fullName || !email || !degree || !specialization) {
      return res.status(400).json({ message: 'All fields are required' });
    }

    const existing = await Therapist.findOne({ username });
    if (existing) {
      return res.status(400).json({ message: 'Username already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newTherapist = new Therapist({
      username,
      password: hashedPassword,
      fullName,
      email,
      degree,
      specialization,
      certificatePath: req.file?.path
    });

    await newTherapist.save();
    res.json({ message: `Therapist ${username} registered successfully!` }); // Added backticks for interpolation
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Server error during registration' });
  }
});

module.exports = router;