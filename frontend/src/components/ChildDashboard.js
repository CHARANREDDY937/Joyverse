"use client";

import { useState, useEffect } from "react";
import "./ChildDashboard.css";

const LEVELS = {
  1: [
    { correct: "cat", scrambled: "tac" },
    { correct: "dog", scrambled: "god" },
    { correct: "bird", scrambled: "drib" },
    { correct: "fish", scrambled: "hsif" },
    { correct: "bear", scrambled: "raeb" },
  ],
  2: [
    { correct: "tiger", scrambled: "tgeir" },
    { correct: "lion", scrambled: "loin" },
    { correct: "monkey", scrambled: "mekony" },
    { correct: "cow", scrambled: "wco" },
    { correct: "buffalo", scrambled: "baffulo" },
  ],
};

export default function ChildDashboard() {
  const [age, setAge] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [isRegistered, setIsRegistered] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [showCelebration, setShowCelebration] = useState(false);
  const [timeLeft, setTimeLeft] = useState(60);
  const [level, setLevel] = useState(1);
  const [gameOver, setGameOver] = useState(false);

  useEffect(() => {
    if (isPlaying && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    } else if (timeLeft === 0) {
      setGameOver(true);
      setIsPlaying(false);
    }
  }, [isPlaying, timeLeft]);

  const handleRegister = () => {
    if (age && name && password) {
      setIsRegistered(true);
    }
  };

  const handleLogin = () => {
    if (name && password) {
      setIsLoggedIn(true);
    }
  };

  const handleStartGame = () => {
    setIsPlaying(true);
    setScore(0);
    setCurrentWordIndex(0);
    setTimeLeft(60);
    setGameOver(false);
  };

  const handleWordChoice = (selectedWord) => {
    if (selectedWord === LEVELS[level][currentWordIndex].correct) {
      setScore((prev) => prev + 1);
      setShowCelebration("Correct! 🌟");
      createStars();
  
      setTimeout(() => {
        setShowCelebration(null);
        if (currentWordIndex < LEVELS[level].length - 1) {
          setCurrentWordIndex((prev) => prev + 1);
        } else {
          setGameOver(true);
          setIsPlaying(false);
        }
      }, 1500);
    } else {
      setShowCelebration("Incorrect ❌");
      
      setTimeout(() => {
        setShowCelebration(null);
        if (currentWordIndex < LEVELS[level].length - 1) {
          setCurrentWordIndex((prev) => prev + 1);
        } else {
          setGameOver(true);
          setIsPlaying(false);
        }
      }, 1500);
    }
  };
  

  const createStars = () => {
    const gameContainer = document.querySelector(".game-container");
    if (gameContainer) {
      for (let i = 0; i < 20; i++) {
        const star = document.createElement("div");
        star.className = "star";
        star.style.left = `${Math.random() * 100}%`;
        star.style.animationDuration = `${Math.random() * 1 + 0.5}s`;
        gameContainer.appendChild(star);
        setTimeout(() => star.remove(), 1500);
      }
    }
  };

  const handleNextLevel = () => {
    setLevel(2);
    handleStartGame();
  };

  return (
    <div className="child-dashboard">
      {!isRegistered ? (
        <div className="register-container">
          <h2>Register Now!</h2>
          <div className="input-group">
            <input type="number" placeholder="Enter your age" value={age} onChange={(e) => setAge(e.target.value)} />
            <input type="text" placeholder="Enter your name" value={name} onChange={(e) => setName(e.target.value)} />
            <input
              type="password"
              placeholder="Create a password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button onClick={handleRegister} className="register-button">
              Register
            </button>
          </div>
        </div>
      ) : !isLoggedIn ? (
        <div className="login-container">
          <h2>Login</h2>
          <div className="input-group">
            <input type="text" placeholder="Enter your name" value={name} onChange={(e) => setName(e.target.value)} />
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <button onClick={handleLogin} className="login-button">
              Start Adventure
            </button>
          </div>
        </div>
      ) : !isPlaying && !gameOver ? (
        <div className="welcome-container">
          <h3>Welcome, {name}!</h3>
          <p>Are you ready to unscramble some words?</p>
          <button onClick={handleStartGame} className="play-button">
            Start Game
          </button>
        </div>
      ) : isPlaying ? (
        <div className="game-container">
          <div className="game-header">
            <div className="score">Score: {score}</div>
            <div className="timer">Time Left: {timeLeft}s</div>
          </div>

          <div className="word-display">
            <h3>Unscramble the word:</h3>
            <div className="scrambled-word">{LEVELS[level][currentWordIndex].scrambled}</div>
          </div>

          <div className="word-choices">
  {currentWordIndex % 2 === 0 ? (
    <>
      <button onClick={() => handleWordChoice(LEVELS[level][currentWordIndex].correct)} className="word-button">
        {LEVELS[level][currentWordIndex].correct}
      </button>
      <button onClick={() => handleWordChoice(LEVELS[level][currentWordIndex].scrambled)} className="word-button">
        {LEVELS[level][currentWordIndex].scrambled}
      </button>
    </>
  ) : (
    <>
      <button onClick={() => handleWordChoice(LEVELS[level][currentWordIndex].scrambled)} className="word-button">
        {LEVELS[level][currentWordIndex].scrambled}
      </button>
      <button onClick={() => handleWordChoice(LEVELS[level][currentWordIndex].correct)} className="word-button">
        {LEVELS[level][currentWordIndex].correct}
      </button>
    </>
  )}
</div>



{showCelebration && <div className="celebration">{showCelebration}</div>}

        </div>
      ) : (
        <div className="game-over">
          <h3>🎉 Great Job, {name}! 🎉</h3>
          <p>You completed this round with a score of {score}!</p>

          {level === 1 ? (
            <>
              <p>You're doing amazing! Want to challenge yourself with Level 2?</p>
              <button onClick={handleNextLevel} className="next-level-button">
                Proceed to Level 2
              </button>
            </>
          ) : (
            <p>You're a word master! Keep up the great work! 🚀</p>
          )}

          <button
            onClick={() => {
              setLevel(1);
              setScore(0);
              setCurrentWordIndex(0);
              setIsPlaying(false);
              setGameOver(false);
            }}
            className="play-again-button"
          >
            Play Again
          </button>
        </div>
      )}
    </div>
  );
}
