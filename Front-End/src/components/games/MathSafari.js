import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './MathSafari.css';

const DIFFICULTY_LEVELS = {
  easy: { max: 10, operations: ['+', '-'] },
  medium: { max: 20, operations: ['+', '-', '×'] },
  hard: { max: 50, operations: ['+', '-', '×', '÷'] }
};

const ANIMALS = {
  elephant: '🐘',
  giraffe: '🦒',
  lion: '🦁',
  monkey: '🐒',
  zebra: '🦓',
  hippo: '🦛'
};

const MathSafari = () => {
  const navigate = useNavigate();
  const [score, setScore] = useState(0);
  const [level, setLevel] = useState('easy');
  const [problem, setProblem] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [streak, setStreak] = useState(0);
  const [currentAnimal, setCurrentAnimal] = useState('elephant');

  const generateProblem = useCallback(() => {
    const { max, operations } = DIFFICULTY_LEVELS[level];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    let num1, num2, answer;

    switch (operation) {
      case '+':
        num1 = Math.floor(Math.random() * max);
        num2 = Math.floor(Math.random() * max);
        answer = num1 + num2;
        break;
      case '-':
        num1 = Math.floor(Math.random() * max);
        num2 = Math.floor(Math.random() * (num1 + 1));
        answer = num1 - num2;
        break;
      case '×':
        num1 = Math.floor(Math.random() * (max / 2));
        num2 = Math.floor(Math.random() * (max / 2));
        answer = num1 * num2;
        break;
      case '÷':
        num2 = Math.floor(Math.random() * (max / 4)) + 1;
        answer = Math.floor(Math.random() * (max / 4)) + 1;
        num1 = num2 * answer;
        break;
      default:
        break;
    }

    setProblem({ num1, num2, operation, answer });
    setUserAnswer('');
    setFeedback(null);
  }, [level]);

  useEffect(() => {
    generateProblem();
  }, [generateProblem]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const numAnswer = parseInt(userAnswer);
    
    if (numAnswer === problem.answer) {
      setFeedback({ correct: true, message: 'Great job! 🎉' });
      setScore(prev => prev + (streak + 1) * 10);
      setStreak(prev => prev + 1);
      
      // Change animal every 5 correct answers
      if ((streak + 1) % 5 === 0) {
        const animals = Object.keys(ANIMALS);
        const currentIndex = animals.indexOf(currentAnimal);
        const nextAnimal = animals[(currentIndex + 1) % animals.length];
        setCurrentAnimal(nextAnimal);
      }

      setTimeout(generateProblem, 1500);
    } else {
      setFeedback({ correct: false, message: 'Try again! 💪' });
      setStreak(0);
    }
  };

  return (
    <div className="math-safari">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Math Safari</h1>
        <div className="game-info">
          <span className="score">Score: {score}</span>
          <span className="streak">Streak: {streak} 🔥</span>
        </div>
      </div>

      <div className="difficulty-selector">
        {Object.keys(DIFFICULTY_LEVELS).map((diff) => (
          <motion.button
            key={diff}
            className={`difficulty-btn ${level === diff ? 'active' : ''}`}
            onClick={() => setLevel(diff)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {diff.charAt(0).toUpperCase() + diff.slice(1)}
          </motion.button>
        ))}
      </div>

      <motion.div 
        className="problem-container"
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
      >
        <div className="animal-container">
          {ANIMALS[currentAnimal]}
        </div>
        
        {problem && (
          <form onSubmit={handleSubmit} className="problem-form">
            <div className="problem">
              <span>{problem.num1}</span>
              <span>{problem.operation}</span>
              <span>{problem.num2}</span>
              <span>=</span>
              <input
                type="number"
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
                placeholder="?"
                autoFocus
              />
            </div>
            <motion.button
              type="submit"
              className="submit-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Check Answer
            </motion.button>
          </form>
        )}

        {feedback && (
          <motion.div
            className={`feedback ${feedback.correct ? 'correct' : 'incorrect'}`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
          >
            {feedback.message}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default MathSafari; 