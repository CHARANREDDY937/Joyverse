// models/Therapist.js
const mongoose = require('mongoose');

const therapistSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  fullName: { type: String, required: true },
  email: { type: String, required: true },
  degree: { type: String, required: true },
  specialization: { type: String, required: true },
  certificatePath: { type: String, required: true },
  isVerified: { type: Boolean, default: false },
  verifiedAt: { type: Date }
});

module.exports = mongoose.model('Therapist', therapistSchema);