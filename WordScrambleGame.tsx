
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { Home, RefreshCw, Check, Clock, Award } from 'lucide-react';
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid';

// Word lists by difficulty level
const wordsByLevel = {
  beginner: ['cat', 'dog', 'sun', 'hat', 'pen', 'cup', 'toy', 'bed', 'lip', 'map'],
  intermediate: ['apple', 'happy', 'child', 'plant', 'water', 'dance', 'smile', 'bread', 'green', 'mouse'],
  expert: ['tomorrow', 'beautiful', 'computer', 'elephant', 'chocolate', 'butterfly', 'adventure', 'universe', 'knowledge', 'wonderful']
};

// Settings for different difficulty levels
const difficultySettings = {
  beginner: {
    wordCount: 5,
    timePerWord: 60, // in seconds
    pointsPerCorrect: 10,
    fastBonus: 5,
    fastTime: 30, // seconds
    completionBonus: 10
  },
  intermediate: {
    wordCount: 8,
    timePerWord: 45,
    pointsPerCorrect: 20,
    fastBonus: 10,
    fastTime: 20,
    completionBonus: 20,
    penaltyPerWrong: 5
  },
  expert: {
    wordCount: 10,
    timePerWord: 30,
    pointsPerCorrect: 30,
    fastBonus: 15,
    fastTime: 10,
    completionBonus: 30,
    penaltyPerWrong: 10
  }
};

const WordScrambleGame: React.FC = () => {
  const navigate = useNavigate();
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('beginner');
  const [gameStarted, setGameStarted] = useState<boolean>(false);
  const [currentWordIndex, setCurrentWordIndex] = useState<number>(0);
  const [words, setWords] = useState<string[]>([]);
  const [scrambledWord, setScrambledWord] = useState<string>('');
  const [userInput, setUserInput] = useState<string>('');
  const [score, setScore] = useState<number>(0);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [timeTaken, setTimeTaken] = useState<number>(0);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [gameSessionId] = useState<string>(uuidv4());
  const [gameStats, setGameStats] = useState<{
    correct: number;
    incorrect: number;
    timeBonus: number;
    completionBonus: number;
  }>({
    correct: 0,
    incorrect: 0,
    timeBonus: 0,
    completionBonus: 0
  });

  // Camera recording refs and state
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);

  // Start camera recording
  const startCameraRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      setCameraStream(stream);
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
    } catch (err) {
      toast.error('Unable to access camera for recording.');
      console.error('Camera error:', err);
    }
  };

  // Stop camera recording
  const stopCameraRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {
        // ignore if already stopped
      }
      mediaRecorderRef.current = null;
    }
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => {
        try {
          track.stop();
        } catch (e) {
          // ignore if already stopped
        }
      });
      setCameraStream(null);
    }
  }, [cameraStream]);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCameraRecording();
    };
  }, [stopCameraRecording]);

  // Select random words based on difficulty
  const selectRandomWords = useCallback(() => {
    const levelWords = wordsByLevel[difficulty];
    const count = difficultySettings[difficulty].wordCount;
    const shuffled = [...levelWords].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }, [difficulty]);

  // Scramble the current word
  const scrambleCurrentWord = useCallback((word: string) => {
    const wordArray = word.split('');
    for (let i = wordArray.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [wordArray[i], wordArray[j]] = [wordArray[j], wordArray[i]];
    }
    const scrambled = wordArray.join('');
    if (scrambled === word && word.length > 1) {
      return scrambleCurrentWord(word);
    }
    return scrambled;
  }, []);

  // Initialize game
  const startGame = useCallback(() => {
    const selectedWords = selectRandomWords();
    setWords(selectedWords);
    setCurrentWordIndex(0);
    setScrambledWord(scrambleCurrentWord(selectedWords[0]));
    setScore(0);
    setTimeLeft(difficultySettings[difficulty].timePerWord);
    setTimeTaken(0);
    setGameStarted(true);
    setIsCorrect(null);
    setGameOver(false);
    setGameStats({
      correct: 0,
      incorrect: 0,
      timeBonus: 0,
      completionBonus: 0
    });
    setUserInput('');
    toast.success('Game started!');
    startCameraRecording();
  }, [difficulty, scrambleCurrentWord, selectRandomWords]);

  // Timer effect
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (gameStarted && !gameOver && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            clearInterval(timer);
            handleTimeUp();
            return 0;
          }
          return prev - 1;
        });
        setTimeTaken(prev => prev + 1);
      }, 1000);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [gameStarted, timeLeft, gameOver]);

  // Handle time up for a word
  const handleTimeUp = () => {
    setIsCorrect(false);
    setGameStats(prev => ({...prev, incorrect: prev.incorrect + 1}));
    setTimeout(() => {
      moveToNextWord();
    }, 1500);
  };

  // Check the user's answer
  const checkAnswer = () => {
    const currentWord = words[currentWordIndex];
    const isAnswerCorrect = userInput.toLowerCase().trim() === currentWord.toLowerCase();
    setIsCorrect(isAnswerCorrect);
    if (isAnswerCorrect) {
      let pointsEarned = difficultySettings[difficulty].pointsPerCorrect;
      const timeSpent = difficultySettings[difficulty].timePerWord - timeLeft;
      let timeBonus = 0;
      if (timeSpent <= difficultySettings[difficulty].fastTime) {
        timeBonus = difficultySettings[difficulty].fastBonus;
        setGameStats(prev => ({
          ...prev, 
          correct: prev.correct + 1,
          timeBonus: prev.timeBonus + timeBonus
        }));
        toast.success(`Fast bonus: +${timeBonus}!`);
      } else {
        setGameStats(prev => ({...prev, correct: prev.correct + 1}));
      }
      setScore(prev => prev + pointsEarned + timeBonus);
      toast.success('Correct!');
      setTimeout(() => {
        moveToNextWord();
      }, 1500);
    } else {
      if (difficulty !== 'beginner' && difficultySettings[difficulty].penaltyPerWrong) {
        setScore(prev => Math.max(0, prev - difficultySettings[difficulty].penaltyPerWrong));
        toast.error(`Wrong answer! -${difficultySettings[difficulty].penaltyPerWrong} points`);
      } else {
        toast.error('Wrong answer! Try again');
      }
      setGameStats(prev => ({...prev, incorrect: prev.incorrect + 1}));
      if (difficulty !== 'beginner') {
        setTimeout(() => {
          moveToNextWord();
        }, 1500);
      }
    }
  };

  // Move to the next word or end game
  const moveToNextWord = () => {
    const nextIndex = currentWordIndex + 1;
    if (nextIndex < words.length) {
      setCurrentWordIndex(nextIndex);
      setScrambledWord(scrambleCurrentWord(words[nextIndex]));
      setTimeLeft(difficultySettings[difficulty].timePerWord);
      setUserInput('');
      setIsCorrect(null);
    } else {
      finishGame(true);
    }
  };

  // Finish the game
  const finishGame = (completed: boolean) => {
    setGameStarted(false);
    setGameOver(true);
    stopCameraRecording();
    if (completed) {
      const completionBonus = difficultySettings[difficulty].completionBonus;
      setScore(prev => prev + completionBonus);
      setGameStats(prev => ({...prev, completionBonus}));
      toast.success(`Game completed! Bonus: +${completionBonus}`);
    }
    const gameSession = {
      id: gameSessionId,
      childId: 'current-child-id', // This would be from the auth context
      gameType: 'word-scramble',
      startTime: new Date().toISOString(),
      endTime: new Date().toISOString(),
      score: score,
      difficulty: difficulty
    };
    console.log('Game session data:', gameSession);
  };

  // Reset the game
  const resetGame = () => {
    setDifficulty('beginner');
    setGameStarted(false);
    setCurrentWordIndex(0);
    setWords([]);
    setScrambledWord('');
    setUserInput('');
    setScore(0);
    setTimeLeft(0);
    setTimeTaken(0);
    setIsCorrect(null);
    setGameOver(false);
    stopCameraRecording();
  };

  // Exit the game
  const exitGame = () => {
    stopCameraRecording();
    navigate('/games');
  };

  // Select difficulty
  const handleDifficultySelect = (level: 'beginner' | 'intermediate' | 'expert') => {
    setDifficulty(level);
  };

  // Handle keyboard input
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && gameStarted && !gameOver && userInput.trim()) {
      checkAnswer();
    }
  };

  return (
    <div className="container mx-auto p-4">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bubblegum text-therapy-vibrant-purple mb-2">
          Word Scramble
        </h1>
        <p className="text-lg">Unscramble the letters to form the correct word!</p>
      </div>

      {!gameStarted && !gameOver ? (
        <div className="flex justify-center items-center min-h-[60vh]">
          <Card className="p-8 max-w-xl w-full mx-auto rounded-3xl shadow-lg">
            <CardContent className="p-2">
              <div className="text-center p-6">
                <h2 className="text-2xl font-bubblegum mb-6">Choose Difficulty</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                  <Button 
                    onClick={() => handleDifficultySelect('beginner')}
                    className={`py-6 ${difficulty === 'beginner' ? 'bg-therapy-green' : 'bg-therapy-gray'}`}
                  >
                    Beginner
                  </Button>
                  <Button 
                    onClick={() => handleDifficultySelect('intermediate')}
                    className={`py-6 ${difficulty === 'intermediate' ? 'bg-therapy-orange' : 'bg-therapy-gray'}`}
                  >
                    Intermediate
                  </Button>
                  <Button 
                    onClick={() => handleDifficultySelect('expert')}
                    className={`py-6 ${difficulty === 'expert' ? 'bg-therapy-purple' : 'bg-therapy-gray'}`}
                  >
                    Expert
                  </Button>
                </div>
                <div className="bg-therapy-gray p-4 rounded-xl mb-6">
                  <h3 className="font-bubblegum mb-2">Difficulty Info</h3>
                  <p className="mb-2">
                    <strong>Beginner:</strong> {difficultySettings.beginner.wordCount} words, {difficultySettings.beginner.timePerWord} seconds each
                  </p>
                  <p className="mb-2">
                    <strong>Intermediate:</strong> {difficultySettings.intermediate.wordCount} words, {difficultySettings.intermediate.timePerWord} seconds each
                  </p>
                  <p>
                    <strong>Expert:</strong> {difficultySettings.expert.wordCount} words, {difficultySettings.expert.timePerWord} seconds each
                  </p>
                </div>
                <Button 
                  onClick={startGame} 
                  className="px-8 py-6 text-xl bg-therapy-vibrant-purple hover:bg-therapy-vibrant-purple/90"
                >
                  Start Game
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 flex justify-center items-center min-h-[500px]">
            <Card className="p-8 max-w-xl w-full mx-auto rounded-3xl shadow-lg">
              <CardContent className="p-2">
                {!gameStarted && !gameOver ? (
                  <div className="text-center p-6">
                    <h2 className="text-2xl font-bubblegum mb-6">Choose Difficulty</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                      <Button 
                        onClick={() => handleDifficultySelect('beginner')}
                        className={`py-6 ${difficulty === 'beginner' ? 'bg-therapy-green' : 'bg-therapy-gray'}`}
                      >
                        Beginner
                      </Button>
                      <Button 
                        onClick={() => handleDifficultySelect('intermediate')}
                        className={`py-6 ${difficulty === 'intermediate' ? 'bg-therapy-orange' : 'bg-therapy-gray'}`}
                      >
                        Intermediate
                      </Button>
                      <Button 
                        onClick={() => handleDifficultySelect('expert')}
                        className={`py-6 ${difficulty === 'expert' ? 'bg-therapy-purple' : 'bg-therapy-gray'}`}
                      >
                        Expert
                      </Button>
                    </div>
                    <div className="bg-therapy-gray p-4 rounded-xl mb-6">
                      <h3 className="font-bubblegum mb-2">Difficulty Info</h3>
                      <p className="mb-2">
                        <strong>Beginner:</strong> {difficultySettings.beginner.wordCount} words, {difficultySettings.beginner.timePerWord} seconds each
                      </p>
                      <p className="mb-2">
                        <strong>Intermediate:</strong> {difficultySettings.intermediate.wordCount} words, {difficultySettings.intermediate.timePerWord} seconds each
                      </p>
                      <p>
                        <strong>Expert:</strong> {difficultySettings.expert.wordCount} words, {difficultySettings.expert.timePerWord} seconds each
                      </p>
                    </div>
                    <Button 
                      onClick={startGame} 
                      className="px-8 py-6 text-xl bg-therapy-vibrant-purple hover:bg-therapy-vibrant-purple/90"
                    >
                      Start Game
                    </Button>
                  </div>
                ) : gameOver ? (
                  <div className="text-center p-6">
                    <h2 className="text-3xl font-bubblegum text-green-600 mb-4">
                      Game Over!
                    </h2>
                    
                    <div className="bg-therapy-gray p-4 rounded-xl mb-6">
                      <h3 className="text-2xl font-bubblegum mb-4">Final Score: {score}</h3>
                      
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="bg-white p-3 rounded-lg">
                          <p className="font-medium">Correct Answers</p>
                          <p className="text-xl text-green-600">{gameStats.correct}</p>
                        </div>
                        <div className="bg-white p-3 rounded-lg">
                          <p className="font-medium">Incorrect Answers</p>
                          <p className="text-xl text-red-600">{gameStats.incorrect}</p>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-white p-3 rounded-lg">
                          <p className="font-medium">Time Bonuses</p>
                          <p className="text-xl text-blue-600">+{gameStats.timeBonus}</p>
                        </div>
                        <div className="bg-white p-3 rounded-lg">
                          <p className="font-medium">Completion Bonus</p>
                          <p className="text-xl text-purple-600">+{gameStats.completionBonus}</p>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex justify-center gap-4">
                      <Button onClick={() => {resetGame(); startGame();}} className="flex items-center gap-2">
                        <RefreshCw className="h-4 w-4" />
                        Play Again
                      </Button>
                      <Button variant="outline" onClick={exitGame} className="flex items-center gap-2">
                        <Home className="h-4 w-4" />
                        Exit Game
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex justify-between items-center mb-4">
                      <div className="flex items-center gap-2">
                        <Award className="h-5 w-5" />
                        <span className="text-lg font-bold">Score: {score}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-5 w-5" />
                        <span className="text-lg font-bold">Time: {timeLeft}s</span>
                      </div>
                      <div className="text-lg">
                        <span className="font-medium">Word:</span> {currentWordIndex + 1}/{words.length}
                      </div>
                    </div>
                    
                    <div className="flex flex-col items-center justify-center py-10">
                      <div className="mb-8 text-center">
                        <h2 className="text-lg text-gray-600 mb-2">Unscramble this word:</h2>
                        <div className="flex justify-center mb-4">
                          {scrambledWord.split('').map((letter, index) => (
                            <div 
                              key={index}
                              className="w-12 h-12 bg-therapy-vibrant-purple text-white text-2xl font-bold flex items-center justify-center rounded-lg m-1"
                            >
                              {letter}
                            </div>
                          ))}
                        </div>
                        
                        {difficulty === 'beginner' && (
                          <div className="text-sm text-gray-500 mb-4">
                            Hint: This word has {words[currentWordIndex]?.length} letters
                          </div>
                        )}
                      </div>
                      
                      <div className="w-full max-w-md mb-6">
                        <input
                          type="text"
                          value={userInput}
                          onChange={(e) => setUserInput(e.target.value)}
                          onKeyDown={handleKeyDown}
                          className={`w-full p-4 text-xl border-2 rounded-lg text-center ${
                            isCorrect === true
                              ? 'border-green-500 bg-green-50'
                              : isCorrect === false
                              ? 'border-red-500 bg-red-50'
                              : 'border-gray-300'
                          }`}
                          placeholder="Type your answer here..."
                          autoFocus
                        />
                      </div>
                      
                      <Button 
                        onClick={checkAnswer} 
                        disabled={!userInput.trim()} 
                        className="flex items-center gap-2 px-8 py-6 text-xl"
                      >
                        <Check className="h-5 w-5" /> Check Answer
                      </Button>
                      
                      {isCorrect === true && (
                        <div className="mt-4 text-green-600 font-bold text-lg animate-bounce">
                          Correct! Well done!
                        </div>
                      )}
                      
                      {isCorrect === false && (
                        <div className="mt-4 text-red-600 font-bold text-lg">
                          {difficulty === 'beginner' ? 'Try again!' : `The correct word was: ${words[currentWordIndex]}`}
                        </div>
                      )}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </div>
          {/* ... right column for emotion detector or other content ... */}
        </div>
      )}
      <div className="flex justify-center mt-6">
        <Button variant="outline" onClick={exitGame} className="flex items-center gap-2">
          <Home className="h-4 w-4" />
          Back to Games
        </Button>
      </div>
    </div>
  );
};

export default WordScrambleGame;
