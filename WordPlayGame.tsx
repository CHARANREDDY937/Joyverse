import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useNavigate, useLocation } from 'react-router-dom';
import { Home, RefreshCw, Check, Clock, Award } from 'lucide-react';
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from '@/context/AuthContext';

// Category groups by difficulty level
const categoriesByLevel = {
  beginner: [
    {
      name: 'Animals',
      words: ['dog', 'cat', 'bird', 'fish'],
      otherWords: ['apple', 'car', 'book', 'sun']
    },
    {
      name: 'Colors',
      words: ['red', 'blue', 'green', 'yellow'],
      otherWords: ['dog', 'hat', 'chair', 'pen']
    },
    {
      name: 'Food',
      words: ['apple', 'bread', 'milk', 'cake'],
      otherWords: ['car', 'dog', 'tree', 'book']
    },
    {
      name: 'Clothes',
      words: ['hat', 'shirt', 'pants', 'shoes'],
      otherWords: ['cat', 'book', 'house', 'tree']
    }
  ],
  intermediate: [
    {
      name: 'Vehicles',
      words: ['car', 'bus', 'train', 'plane', 'boat', 'bike'],
      otherWords: ['apple', 'dog', 'house', 'chair', 'pen', 'book']
    },
    {
      name: 'Fruits',
      words: ['apple', 'banana', 'orange', 'grape', 'peach', 'kiwi'],
      otherWords: ['dog', 'car', 'book', 'shirt', 'chair', 'pen']
    },
    {
      name: 'School Items',
      words: ['book', 'pen', 'paper', 'ruler', 'eraser', 'pencil'],
      otherWords: ['dog', 'apple', 'car', 'shirt', 'sun', 'tree']
    },
    {
      name: 'Emotions',
      words: ['happy', 'sad', 'angry', 'scared', 'excited', 'calm'],
      otherWords: ['apple', 'dog', 'car', 'book', 'tree', 'sun']
    },
    {
      name: 'Weather',
      words: ['rain', 'snow', 'wind', 'sun', 'cloud', 'storm'],
      otherWords: ['apple', 'dog', 'car', 'book', 'pen', 'shirt']
    },
    {
      name: 'Places',
      words: ['home', 'school', 'park', 'store', 'beach', 'zoo'],
      otherWords: ['apple', 'dog', 'car', 'book', 'pen', 'shirt']
    }
  ],
  expert: [
    {
      name: 'Sports',
      words: ['soccer', 'basketball', 'tennis', 'swimming', 'running', 'baseball', 'volleyball', 'hockey', 'football', 'golf'],
      otherWords: ['apple', 'book', 'car', 'house', 'tree', 'pen', 'shirt', 'dog', 'cloud', 'rain']
    },
    {
      name: 'Musical Instruments',
      words: ['piano', 'guitar', 'drum', 'violin', 'flute', 'trumpet', 'saxophone', 'harp', 'clarinet', 'cello'],
      otherWords: ['apple', 'book', 'car', 'house', 'tree', 'pen', 'shirt', 'dog', 'cloud', 'rain']
    },
    {
      name: 'Countries',
      words: ['America', 'Canada', 'Mexico', 'Brazil', 'France', 'Spain', 'China', 'Japan', 'Italy', 'Australia'],
      otherWords: ['apple', 'book', 'car', 'house', 'tree', 'pen', 'shirt', 'dog', 'cloud', 'rain']
    },
    {
      name: 'Occupations',
      words: ['teacher', 'doctor', 'firefighter', 'police', 'chef', 'artist', 'pilot', 'nurse', 'farmer', 'scientist'],
      otherWords: ['apple', 'book', 'car', 'house', 'tree', 'pen', 'shirt', 'dog', 'cloud', 'rain']
    },
    {
      name: 'Body Parts',
      words: ['head', 'arm', 'leg', 'foot', 'hand', 'eye', 'ear', 'nose', 'mouth', 'finger'],
      otherWords: ['apple', 'book', 'car', 'house', 'tree', 'pen', 'shirt', 'dog', 'cloud', 'rain']
    }
  ]
};

// Settings for different difficulty levels
const difficultySettings = {
  beginner: {
    categoryCount: 4,
    wordsPerCategory: 4,
    timePerRound: 60, // in seconds
    pointsPerCorrect: 10,
    fastBonus: 5,
    fastTime: 30, // seconds
    completionBonus: 10
  },
  intermediate: {
    categoryCount: 6,
    wordsPerCategory: 6,
    timePerRound: 45,
    pointsPerCorrect: 20,
    fastBonus: 10,
    fastTime: 20,
    completionBonus: 20,
    penaltyPerWrong: 5
  },
  expert: {
    categoryCount: 10,
    wordsPerCategory: 10,
    timePerRound: 30,
    pointsPerCorrect: 30,
    fastBonus: 15,
    fastTime: 10,
    completionBonus: 30,
    penaltyPerWrong: 10
  }
};

const WordPlayGame: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>(
    location.state?.difficulty || 'beginner'
  );
  const [gameStarted, setGameStarted] = useState<boolean>(false);
  const [currentCategoryIndex, setCurrentCategoryIndex] = useState<number>(0);
  const [categories, setCategories] = useState<any[]>([]);
  const [currentCategory, setCurrentCategory] = useState<any>(null);
  const [options, setOptions] = useState<{ word: string, isCorrect: boolean }[]>([]);
  const [selectedWords, setSelectedWords] = useState<number[]>([]);
  const [score, setScore] = useState<number>(0);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [timeTaken, setTimeTaken] = useState<number>(0);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [gameSessionId] = useState<string>(uuidv4());
  const [emotionLog, setEmotionLog] = useState<{ emotion: string, timestamp: string }[]>([]);
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
  const [submittedAnswers, setSubmittedAnswers] = useState<boolean>(false);
  const [questions, setQuestions] = useState<any[]>([]);

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
      console.log('Camera recording started');
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
    console.log('Camera recording stopped');
  }, [cameraStream]);

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCameraRecording();
    };
  }, [stopCameraRecording]);

  // Select random categories based on difficulty
  const selectRandomCategories = useCallback(() => {
    const levelCategories = categoriesByLevel[difficulty];
    const count = Math.min(difficultySettings[difficulty].categoryCount, levelCategories.length);
    const shuffled = [...levelCategories].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }, [difficulty]);

  // Generate options for the current round
  const generateOptions = useCallback((currentCat: any) => {
    const wordCount = difficultySettings[difficulty].wordsPerCategory;
    const correctWords = currentCat.words.slice(0, wordCount)
      .map(word => ({ word, isCorrect: true }));
    const incorrectWords = currentCat.otherWords.slice(0, wordCount)
      .map(word => ({ word, isCorrect: false }));
    const allOptions = [...correctWords, ...incorrectWords]
      .sort(() => 0.5 - Math.random());
    return allOptions;
  }, [difficulty]);

  // Initialize game
  const startGame = useCallback(() => {
    const selectedCategories = selectRandomCategories();
    setCategories(selectedCategories);
    setCurrentCategoryIndex(0);

    const firstCategory = selectedCategories[0];
    setCurrentCategory(firstCategory);
    setOptions(generateOptions(firstCategory));

    setScore(0);
    setTimeLeft(difficultySettings[difficulty].timePerRound);
    setTimeTaken(0);
    setGameStarted(true);
    setIsCorrect(null);
    setGameOver(false);
    setSelectedWords([]);
    setSubmittedAnswers(false);
    setGameStats({
      correct: 0,
      incorrect: 0,
      timeBonus: 0,
      completionBonus: 0
    });
    setQuestions([]);

    startCameraRecording();
    toast.success('Game started!');
  }, [difficulty, generateOptions, selectRandomCategories]);

  // Timer effect
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (gameStarted && !gameOver && timeLeft > 0 && !submittedAnswers) {
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
  }, [gameStarted, timeLeft, gameOver, submittedAnswers]);

  // Handle time up for a round
  const handleTimeUp = () => {
    setSubmittedAnswers(true);
    checkAnswers();
  };

  // Toggle selection of a word
  const toggleWordSelection = (index: number) => {
    if (submittedAnswers) return;

    if (selectedWords.includes(index)) {
      setSelectedWords(prev => prev.filter(i => i !== index));
    } else {
      setSelectedWords(prev => [...prev, index]);
    }
  };

  // Check answers when submitted
  const checkAnswers = () => {
    setSubmittedAnswers(true);

    let correctSelections = 0;
    let incorrectSelections = 0;

    options.forEach((option, index) => {
      if (selectedWords.includes(index) && option.isCorrect) {
        correctSelections++;
      } else if (selectedWords.includes(index) && !option.isCorrect) {
        incorrectSelections++;
      } else if (!selectedWords.includes(index) && option.isCorrect) {
        incorrectSelections++;
      }
    });

    const totalCorrectOptions = options.filter(o => o.isCorrect).length;
    const isAllCorrect = correctSelections === totalCorrectOptions && incorrectSelections === 0;

    setIsCorrect(isAllCorrect);

    if (isAllCorrect) {
      let pointsEarned = difficultySettings[difficulty].pointsPerCorrect;
      const timeSpent = difficultySettings[difficulty].timePerRound - timeLeft;
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
        setGameStats(prev => ({ ...prev, correct: prev.correct + 1 }));
      }

      setScore(prev => prev + pointsEarned + timeBonus);

      const newQuestion = {
        questionId: currentCategoryIndex + 1,
        answeredCorrectly: true,
        timeTaken: difficultySettings[difficulty].timePerRound - timeLeft
      };

      setQuestions(prev => [...prev, newQuestion]);
      logEmotion('joy');
      toast.success('All correct! Great job!');
    } else {
      if (difficulty !== 'beginner' && difficultySettings[difficulty].penaltyPerWrong) {
        const penalty = difficultySettings[difficulty].penaltyPerWrong * incorrectSelections;
        setScore(prev => Math.max(0, prev - penalty));
        toast.error(`Some answers were wrong! -${penalty} points`);
      } else {
        toast.error('Some answers were wrong!');
      }

      setGameStats(prev => ({ ...prev, incorrect: prev.incorrect + 1 }));

      const newQuestion = {
        questionId: currentCategoryIndex + 1,
        answeredCorrectly: false,
        timeTaken: difficultySettings[difficulty].timePerRound - timeLeft
      };

      setQuestions(prev => [...prev, newQuestion]);
      logEmotion('frustration');
    }

    setTimeout(() => {
      moveToNextCategory();
    }, 3000);
  };

  // Move to the next category or end game
  const moveToNextCategory = () => {
    const nextIndex = currentCategoryIndex + 1;

    if (nextIndex < categories.length) {
      setCurrentCategoryIndex(nextIndex);
      const nextCategory = categories[nextIndex];
      setCurrentCategory(nextCategory);
      setOptions(generateOptions(nextCategory));
      setTimeLeft(difficultySettings[difficulty].timePerRound);
      setSelectedWords([]);
      setSubmittedAnswers(false);
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
      setGameStats(prev => ({ ...prev, completionBonus }));
      toast.success(`Game completed! Bonus: +${completionBonus}`);
      logEmotion('accomplishment');
    }

    const gameSession = {
      id: gameSessionId,
      childId: user?.id || 'guest',
      gameType: 'word-play',
      startTime: new Date().toISOString(),
      endTime: new Date().toISOString(),
      score: score,
      emotionData: emotionLog.map(log => ({
        emotion: log.emotion,
        intensity: Math.random() * 100,
        timestamp: log.timestamp
      })),
      difficulty: difficulty,
      questions: questions
    };

    console.log('Game session data:', gameSession);
  };

  // Log emotion
  const logEmotion = (emotion: string) => {
    const timestamp = new Date().toISOString();
    setEmotionLog(prev => [...prev, { emotion, timestamp }]);
    console.log('Emotion detected:', emotion, 'at', timestamp);
  };

  // Reset the game
  const resetGame = () => {
    setDifficulty('beginner');
    setGameStarted(false);
    setCurrentCategoryIndex(0);
    setCategories([]);
    setCurrentCategory(null);
    setOptions([]);
    setScore(0);
    setTimeLeft(0);
    setTimeTaken(0);
    setIsCorrect(null);
    setGameOver(false);
    setSelectedWords([]);
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

  return (
    <div className="container mx-auto p-4">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bubblegum text-therapy-vibrant-purple mb-2">
          Word Play
        </h1>
        <p className="text-lg">Select all words that belong in the category!</p>
      </div>

      <div className="flex justify-center">
        <Card className="p-4 max-w-md w-full">
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
                    <strong>Beginner:</strong> {difficultySettings.beginner.categoryCount} categories with {difficultySettings.beginner.wordsPerCategory} words each
                  </p>
                  <p className="mb-2">
                    <strong>Intermediate:</strong> {difficultySettings.intermediate.categoryCount} categories with {difficultySettings.intermediate.wordsPerCategory} words each
                  </p>
                  <p>
                    <strong>Expert:</strong> {difficultySettings.expert.categoryCount} categories with {difficultySettings.expert.wordsPerCategory} words each
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
                      <p className="font-medium">Correct Categories</p>
                      <p className="text-xl text-green-600">{gameStats.correct}</p>
                    </div>
                    <div className="bg-white p-3 rounded-lg">
                      <p className="font-medium">Incorrect Categories</p>
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
                    <span className="font-medium">Category:</span> {currentCategoryIndex + 1}/{categories.length}
                  </div>
                </div>
                
                <div className="flex flex-col items-center justify-center py-6">
                  <div className="mb-6 text-center">
                    <h2 className="text-xl text-gray-600 mb-2">Select all words that belong in the category:</h2>
                    <div className="text-3xl font-bold text-therapy-vibrant-purple p-3 bg-therapy-gray rounded-xl mb-6">
                      {currentCategory?.name}
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-6 w-full">
                    {options.map((option, index) => (
                      <div 
                        key={index}
                        onClick={() => toggleWordSelection(index)}
                        className={`p-3 rounded-xl cursor-pointer transition-all text-center
                          ${selectedWords.includes(index) 
                            ? 'bg-therapy-vibrant-purple text-white' 
                            : 'bg-white border-2 border-gray-200 hover:bg-gray-50'}
                          ${submittedAnswers && option.isCorrect 
                            ? 'border-2 border-green-500 bg-green-100' 
                            : submittedAnswers && selectedWords.includes(index) && !option.isCorrect
                            ? 'border-2 border-red-500 bg-red-100'
                            : ''}
                        `}
                      >
                        {option.word}
                      </div>
                    ))}
                  </div>
                  
                  {!submittedAnswers ? (
                    <Button 
                      onClick={checkAnswers} 
                      disabled={selectedWords.length === 0} 
                      className="flex items-center gap-2"
                    >
                      <Check className="h-4 w-4" /> Submit
                    </Button>
                  ) : (
                    <div className="text-center">
                      {isCorrect === true ? (
                        <div className="text-green-600 font-bold text-lg animate-bounce">
                          Perfect! You found all the {currentCategory?.name.toLowerCase()}!
                        </div>
                      ) : (
                        <div className="text-red-600 font-bold text-lg">
                          Not quite. The {currentCategory?.name.toLowerCase()} words are highlighted.
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
      
      <div className="flex justify-center mt-6">
        <Button variant="outline" onClick={exitGame} className="flex items-center gap-2">
          <Home className="h-4 w-4" />
          Back to Games
        </Button>
      </div>
    </div>
  );
};

export default WordPlayGame;