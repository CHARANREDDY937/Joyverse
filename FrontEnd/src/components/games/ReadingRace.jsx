import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './ReadingRace.css';

const STORIES = {
  beginner: [
    {
      id: 1,
      title: "The Friendly Forest",
      text: "In a magical forest, there lived a small rabbit named Hop. Hop loved to make new friends. One day, he met a wise owl, a playful squirrel, and a gentle deer. They all became best friends and had wonderful adventures together.",
      questions: [
        {
          question: "What was the rabbit's name?",
          options: ["Skip", "Hop", "Jump", "Run"],
          correct: "Hop"
        },
        {
          question: "Who did Hop meet in the forest?",
          options: ["A bear and wolf", "An owl, squirrel, and deer", "A fox and rabbit", "A bird and cat"],
          correct: "An owl, squirrel, and deer"
        }
      ]
    },
    {
      id: 2,
      title: "The Magic Garden",
      text: "Sarah found a tiny key in her backyard. It opened a small door in the garden wall. Inside, flowers could talk and butterflies sang sweet songs. It became her favorite secret place.",
      questions: [
        {
          question: "What did Sarah find?",
          options: ["A book", "A toy", "A key", "A flower"],
          correct: "A key"
        },
        {
          question: "What was special about the garden?",
          options: ["It had giant trees", "Flowers could talk and butterflies sang", "It had a pond", "It was very dark"],
          correct: "Flowers could talk and butterflies sang"
        }
      ]
    },
    {
      id: 3,
      title: "The Brave Little Boat",
      text: "A small red boat named Ruby loved sailing on the calm lake. One day, dark clouds brought a storm. Ruby helped rescue three ducks who were caught in the rain. From that day on, everyone called Ruby the bravest boat on the lake.",
      questions: [
        {
          question: "What was the boat's name?",
          options: ["Blue", "Ruby", "Pearl", "Sandy"],
          correct: "Ruby"
        },
        {
          question: "Who did Ruby help during the storm?",
          options: ["Fish", "Ducks", "Swans", "Frogs"],
          correct: "Ducks"
        }
      ]
    }
  ],
  intermediate: [
    {
      id: 4,
      title: "The Time Machine",
      text: "Professor Jane built a special clock in her workshop. When she pressed the golden button, the clock spun backwards and transported her to ancient Egypt! She met pharaohs, saw pyramids being built, and learned how to write in hieroglyphics. But her most amazing discovery was that people from the past weren't so different from people today.",
      questions: [
        {
          question: "What did Professor Jane build?",
          options: ["A robot", "A clock", "A car", "A telephone"],
          correct: "A clock"
        },
        {
          question: "Where did she travel to?",
          options: ["Rome", "China", "Egypt", "Greece"],
          correct: "Egypt"
        },
        {
          question: "What was her most amazing discovery?",
          options: [
            "The pyramids",
            "Hieroglyphics",
            "That people weren't so different",
            "The pharaohs"
          ],
          correct: "That people weren't so different"
        }
      ]
    }
  ],
  advanced: [
    {
      id: 5,
      title: "The Last Tree",
      text: "In the year 2157, twelve-year-old Maya discovered the last real tree on Earth in her great-grandmother's greenhouse. While everyone else lived in a world of artificial plants and digital gardens, Maya learned the ancient art of caring for living things. She collected seeds from the tree and started a movement to bring nature back to the world. Her small garden became the first of many, proving that one person's dedication could change the future.",
      questions: [
        {
          question: "What did Maya find in the greenhouse?",
          options: ["A robot", "The last tree", "A time machine", "A book"],
          correct: "The last tree"
        },
        {
          question: "What did Maya do with the tree's seeds?",
          options: [
            "Ate them",
            "Sold them",
            "Started a movement to bring nature back",
            "Lost them"
          ],
          correct: "Started a movement to bring nature back"
        },
        {
          question: "What was the story's main message?",
          options: [
            "Technology is bad",
            "Trees are important",
            "One person can make a difference",
            "The future is scary"
          ],
          correct: "One person can make a difference"
        }
      ]
    }
  ]
};

const ReadingRace = () => {
  const navigate = useNavigate();
  const [level, setLevel] = useState('beginner');
  const [currentStoryIndex, setCurrentStoryIndex] = useState(0);
  const [showQuestions, setShowQuestions] = useState(false);
  const [score, setScore] = useState(0);
  const [answers, setAnswers] = useState({});
  const [readingStartTime, setReadingStartTime] = useState(null);
  const [readingTime, setReadingTime] = useState(null);
  const [wordsPerMinute, setWordsPerMinute] = useState(null);
  const [error, setError] = useState(null);

  const currentStory = STORIES[level][currentStoryIndex];

  const startReading = useCallback(() => {
    setReadingStartTime(Date.now());
    setShowQuestions(false);
    setAnswers({});
  }, []);

  const calculateReadingSpeed = (text, timeInSeconds) => {
    const wordCount = text.split(/\s+/).length;
    const minutes = timeInSeconds / 60;
    return Math.round(wordCount / minutes);
  };

  const finishReading = () => {
    const endTime = Date.now();
    const timeSpent = Math.floor((endTime - readingStartTime) / 1000);
    setReadingTime(timeSpent);
    const speed = calculateReadingSpeed(currentStory.text, timeSpent);
    setWordsPerMinute(speed);
    setShowQuestions(true);
  };

  const handleAnswer = (questionIndex, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionIndex]: answer
    }));
  };

  const checkAnswers = () => {
    let correct = 0;
    currentStory.questions.forEach((question, index) => {
      if (answers[index] === question.correct) {
        correct++;
      }
    });
    setScore(prev => prev + correct);
    
    // Move to next story or show completion
    if (currentStoryIndex < STORIES[level].length - 1) {
      setCurrentStoryIndex(prev => prev + 1);
      setShowQuestions(false);
      startReading();
    } else {
      // Optionally reset to first story or show a completion message
      setCurrentStoryIndex(0);
      setShowQuestions(false);
      startReading();
    }
  };

  const resetGame = async () => {
    try {
      const response = await fetch('http://localhost:8000/clear_emotions_log', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status !== 'success') {
        setError('Failed to clear emotions log: ' + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError('Error clearing emotions log: ' + error.message);
    }

    setScore(0);
    setCurrentStoryIndex(0);
    setLevel('beginner');
    setShowQuestions(false);
    setAnswers({});
    setReadingTime(null);
    setWordsPerMinute(null);
    startReading();
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
          setError('Failed to clear emotions log: ' + data.message);
        } else {
          setError(null);
        }
      } catch (error) {
        setError('Error clearing emotions log: ' + error.message);
      }

      startReading();
    };

    initializeGame();
  }, [startReading]);

  return (
    <div className="reading-race">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Reading Race</h1>
        <div className="game-info">
          <span className="score">Score: {score}</span>
          {readingTime && (
            <>
              <span className="time">Time: {readingTime}s</span>
              <span className="speed">Speed: {wordsPerMinute} WPM</span>
            </>
          )}
          <motion.button
            className="reset-btn"
            onClick={resetGame}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Reset
          </motion.button>
        </div>
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

      <div className="level-selector">
        {Object.keys(STORIES).map((lvl) => (
          <motion.button
            key={lvl}
            className={`level-btn ${level === lvl ? 'active' : ''}`}
            onClick={() => {
              setLevel(lvl);
              setCurrentStoryIndex(0);
              startReading();
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
          </motion.button>
        ))}
      </div>

      <motion.div 
        className="story-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        {!showQuestions ? (
          <>
            <h2>{currentStory.title}</h2>
            <div className="story-text">
              {currentStory.text}
            </div>
            <motion.button
              className="finish-reading-btn"
              onClick={finishReading}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              I'm Done Reading!
            </motion.button>
          </>
        ) : (
          <div className="questions-container">
            <h3>Answer these questions:</h3>
            {currentStory.questions.map((q, index) => (
              <div key={index} className="question-box">
                <p className="question">{q.question}</p>
                <div className="options-grid">
                  {q.options.map((option, optIndex) => (
                    <motion.button
                      key={optIndex}
                      className={`option-btn ${answers[index] === option ? 'selected' : ''}`}
                      onClick={() => handleAnswer(index, option)}
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      {option}
                    </motion.button>
                  ))}
                </div>
              </div>
            ))}
            <motion.button
              className="check-answers-btn"
              onClick={checkAnswers}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Check Answers
            </motion.button>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default ReadingRace;