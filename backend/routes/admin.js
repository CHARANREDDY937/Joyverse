const express = require('express');
const router = express.Router();
const jwt = require('jsonwebtoken');
const Therapist = require('../models/Therapist');

// Admin login route
router.post('/login', (req, res) => {
  const { username, password } = req.body;
  console.log('Login attempt:', { username, password }); // Debug log

  // Admin credentials
  if (username === 'admin' && password === 'admin123') {
    const token = jwt.sign(
      { id: 'admin', role: 'admin' },
      'your-secret-key',
      { expiresIn: '24h' }
    );

    console.log('Login successful, token generated'); // Debug log
    res.json({
      success: true,
      message: 'Login successful',
      token
    });
  } else {
    console.log('Login failed: invalid credentials'); // Debug log
    res.status(401).json({
      success: false,
      message: 'Invalid credentials'
    });
  }
});

// Get all therapists
router.get('/therapists', async (req, res) => {
  try {
    const therapists = await Therapist.find();
    res.json(therapists);
  } catch (error) {
    console.error('Error fetching therapists:', error);
    res.status(500).json({ message: 'Error fetching therapists' });
  }
});

// Verify therapist route
router.put('/verify-therapist/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const { isVerified } = req.body;
    
    const therapist = await Therapist.findById(id);
    if (!therapist) {
      return res.status(404).json({ message: 'Therapist not found' });
    }

    therapist.isVerified = isVerified;
    therapist.verifiedAt = isVerified ? new Date() : null;
    await therapist.save();

    // Send notification to therapist (implement email service later)
    console.log(`Notification: Therapist ${therapist.fullName} has been ${isVerified ? 'verified' : 'rejected'}`);

    res.status(200).json({ 
      success: true,
      message: `Therapist ${isVerified ? 'verified' : 'rejected'} successfully`, // Added backticks here
      therapist 
    });
  } catch (error) {
    console.error('Error in verification:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Failed to update verification status. Please try again.' 
    });
  }
});

module.exports = router;