import React, { useState } from "react";
import "./MathSafari.css";

const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/b4/42/d8/b442d82da17ab741ba3ff0489e8bd076.jpg',
  Sadness: 'https://i.pinimg.com/736x/e8/0f/79/e80f79c7a4580759dcefdbe3e5e3e186.jpg',
  Anger: 'https://i.pinimg.com/736x/e0/1c/cd/e01ccd8a923eaa6819c53c60eb83d22e.jpg',
  Surprise: 'https://i.pinimg.com/736x/8e/1c/63/8e1c630f0d07d7bf272145c2610f31ff.jpg',
  Disgust: 'https://i.pinimg.com/736x/7e/1f/d9/7e1fd9de9d7b6e59f298f04e6f771206.jpg',
  Fear: 'https://i.pinimg.com/736x/f8/7f/24/f87f245810ff98d211ede79205676d8b.jpg',
  Neutral: 'https://i.pinimg.com/736x/7b/b7/50/7bb750d7fd8f292f2d32565f9ed3f0d6.jpg'
};

const instructionBackground = 'https://i.pinimg.com/736x/ab/80/2b/ab802b4fcaf99c5e79962597fb0f4040.jpg';

export default function MathBalloonGame({ emotion = 'Neutral' }) {
  const [question, setQuestion] = useState({ num1: 8, num2: 1, correct: 9, operator: '+' });
  const [choices, setChoices] = useState([9, 2, 6]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState("");
  const [score, setScore] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isGameStarted, setIsGameStarted] = useState(false);
  const [currentLevel, setCurrentLevel] = useState('EASY');
  const [questionsInLevel, setQuestionsInLevel] = useState(0);

  const generateQuestion = () => {
    let num1 = Math.floor(Math.random() * 10) + 1;
    let num2 = Math.floor(Math.random() * 10) + 1;
    let correct, operator;

    switch (currentLevel) {
      case 'MEDIUM':
        operator = '-';
        if (num1 < num2) [num1, num2] = [num2, num1]; // avoid negative results
        correct = num1 - num2;
        break;
      case 'HARD':
        operator = '*';
        correct = num1 * num2;
        break;
      case 'EXPERT':
        operator = '/';
        correct = num1;
        num1 = num1 * num2;
        break;
      default:
        operator = '+';
        correct = num1 + num2;
    }

    let wrongChoices = [];
    while (wrongChoices.length < 2) {
      const wrong = correct + Math.floor(Math.random() * 10 - 5);
      if (wrong !== correct && !wrongChoices.includes(wrong) && wrong >= 0) {
        wrongChoices.push(wrong);
      }
    }

    const allChoices = [correct, ...wrongChoices].sort(() => Math.random() - 0.5);
    setQuestion({ num1, num2, correct, operator });
    setChoices(allChoices);
  };

  const handleBalloonClick = (value) => {
    if (selected !== null) return;

    setSelected(value);
    setTotalQuestions(prev => prev + 1);
    setQuestionsInLevel(prev => prev + 1);

    if (value === question.correct) {
      setScore(prev => prev + 1);
      setMessage("🎉 Great job! You're a math wizard!");

      if (questionsInLevel + 1 === 3) {
        switch (currentLevel) {
          case 'EASY':
            setCurrentLevel('MEDIUM');
            setMessage("🎉 Level Up! Now try subtraction!");
            break;
          case 'MEDIUM':
            setCurrentLevel('HARD');
            setMessage("🎉 Level Up! Now try multiplication!");
            break;
          case 'HARD':
            setCurrentLevel('EXPERT');
            setMessage("🎉 Level Up! Now try division!");
            break;
          case 'EXPERT':
            setCurrentLevel('EASY');
            setMessage("🎉 Completed all levels! Restarting with addition.");
            break;
          default:
            break;
        }
        setQuestionsInLevel(0);
      }
    } else {
      setMessage("❌ Oops! Try again!");
    }

    setTimeout(() => {
      setSelected(null);
      setMessage("");
      generateQuestion();
    }, 2000);
  };

  const startGame = () => {
    setIsGameStarted(true);
    setScore(0);
    setTotalQuestions(0);
    setCurrentLevel('EASY');
    setQuestionsInLevel(0);
    generateQuestion();
  };

  const backgroundImage = !isGameStarted ? instructionBackground : (emotionBackgrounds[emotion] || emotionBackgrounds.Neutral);

  return (
    <div className="math-game-container" style={{
      backgroundImage: `url(${backgroundImage})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center',
      backgroundRepeat: 'no-repeat'
    }}>
      {!isGameStarted ? (
        <div className="welcome-screen">
          <h1>Welcome to Math Safari!</h1>
          <p>Pop the balloon with the correct answer!</p>
          <button onClick={startGame} className="start-btn">Start Game</button>
        </div>
      ) : (
        <>
          <div className="wooden-plank">
            <div  className="wooden-texture"></div>
            <div className="question-box">
              {question.num1} {question.operator} {question.num2} = {selected !== null ? selected : "?"}
            </div>
          </div>

          <div className="balloon-container">
            {choices.map((choice, i) => (
              <div
                key={i}
                className={`candy-balloon ${selected === choice ? "float" : ""}`}
                onClick={() => handleBalloonClick(choice)}
              >
                {choice}
              </div>
            ))}
          </div>

          {message && <div className="feedback-message">{message}</div>}

          <div className="progress-bar">
            <span className="progress-text">
              Score: {score}/{totalQuestions}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
