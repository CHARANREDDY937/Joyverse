// models/Child.js
const mongoose = require('mongoose');

const childSchema = new mongoose.Schema({
  username: { type: String, required: true }, // Username of the child
  password: { type: String, required: true }, // Password for authentication
  name: { type: String, required: true },     // Full name of the child
});

module.exports = mongoose.model('child', childSchema);