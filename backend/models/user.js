const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  name: { type: String, required: true },
  emotionSummary: [
    {
      game: { type: String, required: true },
      emotion: {
        Anger: { type: Number, default: 0 },
        Sadness: { type: Number, default: 0 },
        Happiness: { type: Number, default: 0 },
        Fear: { type: Number, default: 0 },
        Disgust: { type: Number, default: 0 },
        Surprise: { type: Number, default: 0 },
        Neutral: { type: Number, default: 0 }
      },
      timestamp: { type: Date, required: true }
    }
  ]
});

const User = mongoose.model('User', userSchema);

module.exports = User;