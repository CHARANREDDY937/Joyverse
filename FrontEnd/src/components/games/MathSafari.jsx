import React, { useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import "./MathSafari.css";

const emotionBackgrounds = {
  Happiness: "https://i.pinimg.com/736x/b4/42/d8/b442d82da17ab741ba3ff0489e8bd076.jpg",
  Sadness: "https://i.pinimg.com/736x/e8/0f/79/e80f79c7a4580759dcefdbe3e5e3e186.jpg",
  Anger: "https://i.pinimg.com/736x/e0/1c/cd/e01ccd8a923eaa6819c53c60eb83d22e.jpg",
  Surprise: "https://i.pinimg.com/736x/8e/1c/63/8e1c630f0d07d7bf272145c2610f31ff.jpg",
  Disgust: "https://i.pinimg.com/736x/7e/1f/d9/7e1fd9de9d7b6e59f298f04e6f771206.jpg",
  Fear: "https://i.pinimg.com/736x/f8/7f/24/f87f245810ff98d211ede79205676d8b.jpg",
  Neutral: "https://i.pinimg.com/736x/7b/b7/50/7bb750d7fd8f292f2d32565f9ed3f0d6.jpg",
};

const instructionBackground = "https://i.pinimg.com/736x/ab/80/2b/ab802b4fcaf99c5e79962597fb0f4040.jpg";

export default function MathBalloonGame({ emotion = "Neutral" }) {
  const navigate = useNavigate();
  const [question, setQuestion] = useState(null);
  const [choices, setChoices] = useState([]);
  const [selected, setSelected] = useState(null);
  const [message, setMessage] = useState("");
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [isGameStarted, setIsGameStarted] = useState(false);
  const [currentLevel, setCurrentLevel] = useState("EASY");
  const [questionsInLevel, setQuestionsInLevel] = useState(0);
  const [error, setError] = useState(null);

  const generateQuestion = useCallback(() => {
    let num1 = Math.floor(Math.random() * 10) + 1;
    let num2 = Math.floor(Math.random() * 10) + 1;
    let correct, operator;

    switch (currentLevel) {
      case "MEDIUM":
        operator = "-";
        if (num1 < num2) [num1, num2] = [num2, num1];
        correct = num1 - num2;
        break;
      case "HARD":
        operator = "*";
        correct = num1 * num2;
        break;
      case "EXPERT":
        operator = "/";
        correct = num1;
        num1 = num1 * num2;
        break;
      default:
        operator = "+";
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
    setSelected(null);
    setMessage("");
  }, [currentLevel]);

  const appendEmotionPercentages = async () => {
    try {
      const response = await fetch("http://localhost:8000/append_emotion_percentages", {
        method: "POST",
      });
      const data = await response.json();
      if (data.status !== "success") {
        setError("Failed to append emotion percentages: " + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError("Error appending emotion percentages: " + error.message);
    }
  };

  useEffect(() => {
    if (isGameStarted) {
      generateQuestion();
    }
    return () => {
      appendEmotionPercentages();
      console.log("MathBalloonGame unmounted: Emotion percentages appended.");
    };
  }, [isGameStarted, generateQuestion]);

  const handleBalloonClick = (value) => {
    if (selected !== null) return;

    setSelected(value);
    setTotalQuestions((prev) => prev + 1);
    setQuestionsInLevel((prev) => prev + 1);

    if (value === question.correct) {
      setScore((prev) => prev + (streak + 1) * 10);
      setStreak((prev) => prev + 1);
      setMessage("🎉 Great job! You're a math wizard!");

      if (questionsInLevel + 1 === 3) {
        setCurrentLevel((prev) =>
          prev === "EXPERT" ? "EASY" : prev === "HARD" ? "EXPERT" : prev === "MEDIUM" ? "HARD" : "MEDIUM"
        );
        setMessage("🎉 Level Up! Get ready for more challenges!");
        setQuestionsInLevel(0);
      }
    } else {
      setMessage("❌ Oops! Try again!");
      setStreak(0);
    }

    setTimeout(generateQuestion, 2000);
  };

  const startGame = () => {
    setIsGameStarted(true);
    setScore(0);
    setTotalQuestions(0);
    setCurrentLevel("EASY");
    setQuestionsInLevel(0);
    setStreak(0);
    generateQuestion();
  };

  const handleBack = async () => {
    await appendEmotionPercentages();
    navigate("/child/games");
  };

  const backgroundImage = !isGameStarted ? instructionBackground : emotionBackgrounds[emotion] || emotionBackgrounds.Neutral;

  return (
    <div className="math-game-container" style={{ backgroundImage: `url(${backgroundImage})`, backgroundSize: "cover" }}>
      {!isGameStarted ? (
        <motion.div className="welcome-screen" initial={{ scale: 0.8 }} animate={{ scale: 1 }}>
          <h1>Welcome to Math Safari!</h1>
          <p>Pop the balloon with the correct answer!</p>
          <motion.button onClick={startGame} className="start-btn" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            Start Game
          </motion.button>
        </motion.div>
      ) : (
        <>
          <motion.button className="back-button" onClick={handleBack} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            ← Back to Games
          </motion.button>
          <div className="wooden-plank">
            <div  className="wooden-texture"></div>
            <div className="question-box">
              {question.num1} {question.operator} {question.num2} = {selected !== null ? selected : "?"}
            </div>
          </div>

          <div className="balloon-container">
            {choices.map((choice, i) => (
              <motion.div key={i} className="candy-balloon" onClick={() => handleBalloonClick(choice)} whileHover={{ scale: 1.1 }}>
                {choice}
              </motion.div>
            ))}
          </div>
          {message && <div className="feedback-message">{message}</div>}
        </>
      )}
    </div>
  );
}