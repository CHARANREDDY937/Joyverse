// models/user.js
const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String, required: true },
  emotionSummary: [
    {
      game: { type: String, required: true },
      emotion: {
        Sadness: { type: Number, required: true },
        Happiness: { type: Number, required: true },
        Fear: { type: Number, required: true },
        Disgust: { type: Number, required: true },
        Surprise: { type: Number, required: true },
        Neutral: { type: Number, required: true }
      },
      timestamp: { type: Date, required: true }
    }
  ]
});

const User = mongoose.model('User', userSchema);

module.exports = User;
