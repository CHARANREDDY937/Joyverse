import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './SpellingBee.css';

const WORD_LISTS = {
  easy: [
    { word: 'cat', hint: '🐱 A furry pet that meows' },
    { word: 'dog', hint: '🐕 A loyal furry friend' },
    { word: 'sun', hint: '☀️ Shines in the sky' },
    { word: 'hat', hint: '🎩 Wear it on your head' },
    { word: 'bee', hint: '🐝 Makes honey' },
    { word: 'pig', hint: '🐷 Pink farm animal' },
    { word: 'fox', hint: '🦊 Clever wild animal' },
    { word: 'cow', hint: '🐮 Gives us milk' },
    { word: 'bed', hint: '🛏️ Sleep here' },
    { word: 'cup', hint: '☕ Drink from it' },
    { word: 'car', hint: '🚗 Drive in this' },
    { word: 'pen', hint: '✒️ Write with it' },
    { word: 'box', hint: '📦 Store things inside' },
    { word: 'belt', hint: '👖 Holds up pants' },
    { word: 'key', hint: '🔑 Opens doors' },
    { word: 'fan', hint: '💨 Keeps you cool' },
    { word: 'bus', hint: '🚌 Big vehicle' },
    { word: 'egg', hint: '🥚 Bird lays this' },
    { word: 'ice', hint: '🧊 Frozen water' },
    { word: 'jam', hint: '🍯 Sweet spread' },
    { word: 'kite', hint: '🪁 Flies in the sky' },
    { word: 'moon', hint: '🌙 Shines at night' },
    { word: 'star', hint: '⭐ Twinkles at night' },
    { word: 'fish', hint: '🐠 Swims in water' },
    { word: 'bird', hint: '🐦 Has wings' }
  ],
  medium: [
    { word: 'apple', hint: '🍎 A sweet red fruit' },
    { word: 'beach', hint: '🏖️ Sandy shore by the ocean' },
    { word: 'cloud', hint: '☁️ Float in the sky' },
    { word: 'tiger', hint: '🐯 Striped big cat' },
    { word: 'house', hint: '🏠 Place to live' },
    { word: 'pizza', hint: '🍕 Round food with toppings' },
    { word: 'snake', hint: '🐍 Slithers on ground' },
    { word: 'train', hint: '🚂 Runs on tracks' },
    { word: 'phone', hint: '📱 Call people with it' },
    { word: 'bread', hint: '🍞 Baked food' },
    { word: 'chair', hint: '🪑 Sit on it' },
    { word: 'clock', hint: '⏰ Tells time' },
    { word: 'dress', hint: '👗 Girls wear it' },
    { word: 'flower', hint: '🌸 Grows in garden' },
    { word: 'ghost', hint: '👻 Spooky spirit' },
    { word: 'heart', hint: '❤️ Symbol of love' },
    { word: 'juice', hint: '🧃 Fruit drink' },
    { word: 'lemon', hint: '🍋 Sour fruit' },
    { word: 'money', hint: '💵 Buy things with it' },
    { word: 'music', hint: '🎵 Listen to it' },
    { word: 'ocean', hint: '🌊 Big water body' },
    { word: 'queen', hint: '👑 Female ruler' },
    { word: 'robot', hint: '🤖 Machine person' },
    { word: 'sheep', hint: '🐑 Gives us wool' },
    { word: 'water', hint: '💧 Drink this' }
  ],
  hard: [
    { word: 'elephant', hint: '🐘 Large gray animal with a trunk' },
    { word: 'butterfly', hint: '🦋 Colorful flying insect' },
    { word: 'rainbow', hint: '🌈 Colorful arc in the sky' },
    { word: 'penguin', hint: '🐧 Bird that swims but cannot fly' },
    { word: 'dolphin', hint: '🐬 Smart sea mammal' },
    { word: 'airplane', hint: '✈️ Flies in the sky' },
    { word: 'birthday', hint: '🎂 Celebration day' },
    { word: 'computer', hint: '💻 Electronic device' },
    { word: 'dinosaur', hint: '🦖 Ancient reptile' },
    { word: 'football', hint: '⚽ Sport with ball' },
    { word: 'giraffe', hint: '🦒 Long neck animal' },
    { word: 'hospital', hint: '🏥 Where doctors work' },
    { word: 'icecream', hint: '🍦 Cold sweet treat' },
    { word: 'kangaroo', hint: '🦘 Jumps with pouch' },
    { word: 'mountain', hint: '⛰️ Very high land' },
    { word: 'necklace', hint: '📿 Neck jewelry' },
    { word: 'octopus', hint: '🐙 Eight arms sea animal' },
    { word: 'princess', hint: '👸 Royal daughter' },
    { word: 'question', hint: '❓ Ask this' },
    { word: 'sandwich', hint: '🥪 Food between bread' },
    { word: 'treasure', hint: '💎 Valuable things' },
    { word: 'umbrella', hint: '☔ Use in rain' },
    { word: 'volcano', hint: '🌋 Mountain that erupts' },
    { word: 'whistle', hint: '🎵 Make sound with mouth' },
    { word: 'zebra', hint: '🦓 Black and white stripes' }
  ]
};

const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/13/1b/b2/131bb271fb3a62655937fb287f3825f6.jpg',
  Sadness: 'https://i.pinimg.com/736x/12/39/19/1239197f1336b002089811206dd8d88c.jpg',
  Anger: 'https://i.pinimg.com/736x/c4/67/78/c4677817536412907f44b505c8778058.jpg',
  Surprise: 'https://i.pinimg.com/736x/07/27/56/0727562762212107eb9e07b72a0ad715.jpg',
  Disgust: 'https://i.pinimg.com/736x/59/37/1a/59371a527e75a780cc801e56a61c9965.jpg',
  Fear: 'https://i.pinimg.com/736x/72/bf/ac/72bfac75ef2fa306eb6df57bb57137d5.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const instructionBackground = 'https://i.pinimg.com/736x/08/37/49/083749cab67b130f711b1be18b0428ae.jpg';

const getLevelIndexFromEmotion = (emotion) => {
  switch (emotion) {
    case 'Sadness':
      return 0; // Easy
    case 'Neutral':
      return 1; // Medium
    case 'Happiness':
    case 'Angry':
    case 'Fear':
    case 'Disgust':
      return 2; // Hard
    default:
      return 1; // Fallback to medium
  }
};

const SpellingBee = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [currentWord, setCurrentWord] = useState(null);
  const [userInput, setUserInput] = useState('');
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);
  const [showInstructions, setShowInstructions] = useState(true);
  const [currentLevelIndex, setCurrentLevelIndex] = useState(0);
  const [pendingLevelIndex, setPendingLevelIndex] = useState(getLevelIndexFromEmotion(emotion));
  const [wordsPerLevel, setWordsPerLevel] = useState(3);
  const [currentLevelProgress, setCurrentLevelProgress] = useState(0);

  // Update pendingLevelIndex when emotion changes
  useEffect(() => {
    setPendingLevelIndex(getLevelIndexFromEmotion(emotion));
  }, [emotion]);

  const selectNewWord = useCallback(() => {
    const levels = ['easy', 'medium', 'hard'];
    const currentLevel = levels[currentLevelIndex % 3];
    const words = WORD_LISTS[currentLevel];
    
    // Calculate the starting index for the current group of words
    const baseIndex = Math.floor(currentLevelProgress / wordsPerLevel) * wordsPerLevel;
    const availableWords = words.slice(baseIndex, baseIndex + wordsPerLevel);
    
    // Select a random word from the current group
    const randomWord = availableWords[Math.floor(Math.random() * availableWords.length)];
    setCurrentWord(randomWord);
    setUserInput('');
    setShowHint(false);
    setFeedback(null);
  }, [currentLevelIndex, currentLevelProgress, wordsPerLevel]);

  useEffect(() => {
    selectNewWord();
  }, [selectNewWord]);

  const speakWord = () => {
    const utterance = new SpeechSynthesisUtterance(currentWord.word);
    utterance.rate = 0.6; // Very slow and clear
    utterance.pitch = 1.5; // Higher, more child-like pitch
    window.speechSynthesis.speak(utterance);
  };

  // Ensure voices are loaded
  useEffect(() => {
    window.speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (userInput.toLowerCase() === currentWord.word.toLowerCase()) {
      setFeedback({ type: 'success', message: 'Correct! 🎉' });
      setScore(prev => prev + (streak + 1) * 10);
      setStreak(prev => prev + 1);
      
      // Update level progress
      setCurrentLevelProgress(prev => {
        const newProgress = prev + 1;
        if (newProgress % wordsPerLevel === 0) {
          // Move to next level after completing wordsPerLevel words
          setCurrentLevelIndex(pendingLevelIndex);
          return 0;
        }
        return newProgress;
      });
      
      setTimeout(() => {
        selectNewWord();
      }, 1500);
    } else {
      setFeedback({ type: 'error', message: 'Try again! 💪' });
      setStreak(0);
      // Update level index after incorrect guess
      setCurrentLevelIndex(pendingLevelIndex);
      setTimeout(() => {
        selectNewWord();
      }, 1500);
    }
  };

  const resetGame = async () => {
    // try {
    //   const response = await fetch('http://localhost:8000/clear_emotions_log', {
    //     method: 'POST',
    //   });
    //   const data = await response.json();
    //   if (data.status !== 'success') {
    //     console.error('Failed to clear emotions log:', data.message); // Log to console instead of setting error
    //   }
    // } catch (error) {
    //   console.error('Error clearing emotions log:', error.message); // Log to console instead of setting error
    // }

    setCurrentLevelIndex(pendingLevelIndex);
    setCurrentLevelProgress(0);
    setScore(0);
    setStreak(0);
    setUserInput('');
    setShowHint(false);
    setFeedback(null);
    selectNewWord();
  };

  // Initialize game and clear emotions log on mount
  useEffect(() => {
    const initializeGame = async () => {
      try {
        const response = await fetch('http://localhost:8000/clear_emotions_log', {
          method: 'POST',
        });
        const data = await response.json();
        if (data.status !== 'success') {
          console.error('Failed to clear emotions log:', data.message); // Log to console instead of setting error
        }
      } catch (error) {
        console.error('Error clearing emotions log:', error.message); // Log to console instead of setting error
      }
    };

    initializeGame();

    return () => {
      console.log('SpellingBee unmounted: Emotion logging should stop.');
    };
  }, []);

  const renderInstructions = () => (
    <motion.div 
      className="instructions"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <h1>Welcome to Spelling Bee!</h1>
      <p>🎯 Listen to the word and spell it correctly.</p>
      <p>💡 Use hints when you need help!</p>
      <p>🏆 Score points and build your streak!</p>
      <p>📚 The game adjusts difficulty based on your emotions after each word.</p>
      <motion.button
        className="start-btn"
        onClick={() => setShowInstructions(false)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Start Game
      </motion.button>
    </motion.div>
  );

  return (
    <div 
      className="spelling-bee"
      style={{
        backgroundImage: `url(${showInstructions ? instructionBackground : emotionBackgrounds[emotion]})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        minHeight: '100vh',
        transition: 'background-image 0.5s ease-in-out'
      }}
    >
      {showInstructions ? (
        renderInstructions()
      ) : (
        <>
          <motion.button
            className="back-button"
            onClick={() => navigate('/child/games')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Games
          </motion.button>

          <div className="transparent-box" style={{
            background: 'transparent',
            backdropFilter: 'blur(10px)',
            padding: '20px',
            borderRadius: '15px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            marginTop: '40px',
          }}>
            <h1>Spelling Bee</h1>
            <div className="gamescore">
              <span className="score">Score: {score}</span>
              <span className="streak">Streak: {streak} 🔥</span>
              <motion.button
                className="reset-btn"
                onClick={resetGame}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                Reset
              </motion.button>
            </div>

            {error && (
              <motion.p
                className="error-message"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {error}
              </motion.p>
            )}

            <div className="level-indicator">
              Level: {['Easy', 'Medium', 'Hard'][currentLevelIndex % 3]} 
              ({currentLevelProgress + 1}/{wordsPerLevel})
            </div>
            {currentWord && (
              <motion.div 
                className="game-container"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                <div className="word-controls">
                  <motion.button
                    className="speak-btn"
                    onClick={speakWord}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    🔊 Hear Word
                  </motion.button>
                  <motion.button
                    className="hint-btn"
                    onClick={() => setShowHint(true)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    disabled={showHint}
                  >
                    💡 Show Hint
                  </motion.button>
                </div>

                {showHint && (
                  <motion.div 
                    className="hint"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    {currentWord.hint}
                  </motion.div>
                )}

                <form onSubmit={handleSubmit} className="spelling-form">
                  <input
                    type="text"
                    value={userInput}
                    onChange={(e) => setUserInput(e.target.value)}
                    placeholder="Type the word here..."
                    autoFocus
                  />
                  <motion.button
                    type="submit"
                    className="submit-btn"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Check Spelling
                  </motion.button>
                </form>

                {feedback && (
                  <motion.div
                    className={`feedback ${feedback.type}`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                  >
                    {feedback.message}
                  </motion.div>
                )}
              </motion.div>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default SpellingBee;
