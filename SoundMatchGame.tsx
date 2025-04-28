import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { Home, RefreshCw, Music, Volume2, Clock, Award } from 'lucide-react';
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid';

// Sound items by difficulty level
const soundItemsByLevel = {
  beginner: [
    { id: 1, name: 'dog', soundName: 'bark', image: '🐕' },
    { id: 2, name: 'cat', soundName: 'meow', image: '🐈' },
    { id: 3, name: 'cow', soundName: 'moo', image: '🐄' },
    { id: 4, name: 'bird', soundName: 'chirp', image: '🐦' },
    { id: 5, name: 'duck', soundName: 'quack', image: '🦆' },
    { id: 6, name: 'horse', soundName: 'neigh', image: '🐎' },
    { id: 7, name: 'frog', soundName: 'croak', image: '🐸' },
    { id: 8, name: 'pig', soundName: 'oink', image: '🐖' },
  ],
  intermediate: [
    { id: 1, name: 'violin', soundName: 'violin', image: '🎻' },
    { id: 2, name: 'piano', soundName: 'piano', image: '🎹' },
    { id: 3, name: 'trumpet', soundName: 'trumpet', image: '🎺' },
    { id: 4, name: 'drum', soundName: 'drum', image: '🥁' },
    { id: 5, name: 'guitar', soundName: 'guitar', image: '🎸' },
    { id: 6, name: 'tambourine', soundName: 'tambourine', image: '🪘' },
    { id: 7, name: 'bell', soundName: 'bell', image: '🔔' },
    { id: 8, name: 'flute', soundName: 'flute', image: '🎵' },
    { id: 9, name: 'saxophone', soundName: 'saxophone', image: '🎷' },
    { id: 10, name: 'harp', soundName: 'harp', image: '🪕' },
  ],
  expert: [
    { id: 1, name: 'thunder', soundName: 'thunder', image: '⚡' },
    { id: 2, name: 'rain', soundName: 'rain', image: '🌧️' },
    { id: 3, name: 'wind', soundName: 'wind', image: '💨' },
    { id: 4, name: 'ocean waves', soundName: 'waves', image: '🌊' },
    { id: 5, name: 'doorbell', soundName: 'doorbell', image: '🚪' },
    { id: 6, name: 'car horn', soundName: 'car-horn', image: '🚗' },
    { id: 7, name: 'alarm clock', soundName: 'alarm', image: '⏰' },
    { id: 8, name: 'phone ring', soundName: 'phone', image: '📱' },
    { id: 9, name: 'helicopter', soundName: 'helicopter', image: '🚁' },
    { id: 10, name: 'train', soundName: 'train', image: '🚂' },
    { id: 11, name: 'typewriter', soundName: 'typewriter', image: '⌨️' },
    { id: 12, name: 'camera shutter', soundName: 'camera', image: '📸' },
  ]
};

// Settings for different difficulty levels
const difficultySettings = {
  beginner: {
    itemCount: 5,
    timePerItem: 60,
    optionsPerItem: 3,
    pointsPerCorrect: 10,
    fastBonus: 5,
    fastTime: 30,
    completionBonus: 10
  },
  intermediate: {
    itemCount: 8,
    timePerItem: 45,
    optionsPerItem: 4,
    pointsPerCorrect: 20,
    fastBonus: 10,
    fastTime: 20,
    completionBonus: 20,
    penaltyPerWrong: 5
  },
  expert: {
    itemCount: 10,
    timePerItem: 30,
    optionsPerItem: 5,
    pointsPerCorrect: 30,
    fastBonus: 15,
    fastTime: 10,
    completionBonus: 30,
    penaltyPerWrong: 10
  }
};

const SoundMatchGame: React.FC = () => {
  const navigate = useNavigate();
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('beginner');
  const [gameStarted, setGameStarted] = useState<boolean>(false);
  const [currentRoundIndex, setCurrentRoundIndex] = useState<number>(0);
  const [soundItems, setSoundItems] = useState<any[]>([]);
  const [currentSound, setCurrentSound] = useState<any>(null);
  const [options, setOptions] = useState<any[]>([]);
  const [selectedOption, setSelectedOption] = useState<number | null>(null);
  const [score, setScore] = useState<number>(0);
  const [timeLeft, setTimeLeft] = useState<number>(0);
  const [timeTaken, setTimeTaken] = useState<number>(0);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [gameOver, setGameOver] = useState<boolean>(false);
  const [gameSessionId] = useState<string>(uuidv4());
  const [gameStats, setGameStats] = useState({
    correct: 0,
    incorrect: 0,
    timeBonus: 0,
    completionBonus: 0
  });

  // Refs for media recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  // Select random sound items based on difficulty
  const selectRandomItems = useCallback(() => {
    const levelItems = soundItemsByLevel[difficulty];
    const count = difficultySettings[difficulty].itemCount;
    const shuffled = [...levelItems].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }, [difficulty]);

  // Generate options for the current round
  const generateOptions = useCallback((currentItem: any, allItems: any[]) => {
    const options = [currentItem];
    const incorrectOptions = allItems.filter(item => item.id !== currentItem.id);
    const shuffledIncorrect = [...incorrectOptions].sort(() => 0.5 - Math.random());
    const optionsNeeded = difficultySettings[difficulty].optionsPerItem - 1;
    options.push(...shuffledIncorrect.slice(0, optionsNeeded));
    return options.sort(() => 0.5 - Math.random());
  }, [difficulty]);

  // Start camera recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: true 
      });
      streamRef.current = stream;
      
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recordedChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        console.log('Recording saved:', blob);
      };

      recorder.start(1000);
      console.log('Recording started');
    } catch (error) {
      console.error('Error accessing camera:', error);
      toast.error('Could not access camera for recording');
    }
  };

  // Stop camera recording and clean up
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      console.log('Recording stopped');
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    
    mediaRecorderRef.current = null;
    recordedChunksRef.current = [];
  };

  // Clean up recording when component unmounts
  useEffect(() => {
    return () => {
      stopRecording();
    };
  }, []);

  // Initialize game
  const startGame = useCallback(() => {
    const selectedItems = selectRandomItems();
    setSoundItems(selectedItems);
    setCurrentRoundIndex(0);
    
    const firstItem = selectedItems[0];
    setCurrentSound(firstItem);
    setOptions(generateOptions(firstItem, soundItemsByLevel[difficulty]));
    
    setScore(0);
    setTimeLeft(difficultySettings[difficulty].timePerItem);
    setTimeTaken(0);
    setGameStarted(true);
    setIsCorrect(null);
    setGameOver(false);
    setSelectedOption(null);
    setGameStats({
      correct: 0,
      incorrect: 0,
      timeBonus: 0,
      completionBonus: 0
    });
    
    // Start camera recording
    startRecording();
    
    toast.success('Game started!');
    playSound(firstItem);
  }, [difficulty, generateOptions, selectRandomItems]);

  // Simulate playing a sound
  // Add audio ref for better control
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Modified playSound function with better error handling
  const playSound = (soundItem: any) => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
    
    try {
      const audio = new Audio(`${process.env.PUBLIC_URL}/sounds/${soundItem.soundName}.mp3`);
      audioRef.current = audio;
      
      audio.onerror = (error) => {
        console.error('Audio error:', error);
        toast.error(`Failed to load sound: ${soundItem.soundName}`);
      };

      audio.play().catch((error) => {
        console.error('Playback error:', error);
        toast.error('Click the play button to start audio');
      });
      
      toast.info(`Playing: ${soundItem.soundName}`);
    } catch (error) {
      console.error('Sound initialization error:', error);
      toast.error('Audio system unavailable');
    }
  };

  // Enhanced cleanup in useEffect
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      stopRecording();
    };
  }, []);

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

  // Handle time up for a round
  const handleTimeUp = () => {
    setIsCorrect(false);
    setGameStats(prev => ({...prev, incorrect: prev.incorrect + 1}));
    
    setTimeout(() => {
      moveToNextRound();
    }, 1500);
  };

  // Check the user's answer
  const checkAnswer = (optionIndex: number) => {
    setSelectedOption(optionIndex);
    const selectedItem = options[optionIndex];
    const isAnswerCorrect = selectedItem.id === currentSound.id;
    setIsCorrect(isAnswerCorrect);
    
    if (isAnswerCorrect) {
      let pointsEarned = difficultySettings[difficulty].pointsPerCorrect;
      const timeSpent = difficultySettings[difficulty].timePerItem - timeLeft;
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
        moveToNextRound();
      }, 1500);
    } else {
      if (difficulty !== 'beginner' && difficultySettings[difficulty].penaltyPerWrong) {
        setScore(prev => Math.max(0, prev - difficultySettings[difficulty].penaltyPerWrong));
        toast.error(`Wrong choice! -${difficultySettings[difficulty].penaltyPerWrong} points`);
      } else {
        toast.error('Wrong choice! Try again');
      }
      
      setGameStats(prev => ({...prev, incorrect: prev.incorrect + 1}));
      
      if (difficulty !== 'beginner') {
        setTimeout(() => {
          moveToNextRound();
        }, 1500);
      }
    }
  };

  // Move to the next round or end game
  const moveToNextRound = () => {
    const nextIndex = currentRoundIndex + 1;
    
    if (nextIndex < soundItems.length) {
      setCurrentRoundIndex(nextIndex);
      const nextItem = soundItems[nextIndex];
      setCurrentSound(nextItem);
      setOptions(generateOptions(nextItem, soundItemsByLevel[difficulty]));
      setTimeLeft(difficultySettings[difficulty].timePerItem);
      setSelectedOption(null);
      setIsCorrect(null);
      playSound(nextItem);
    } else {
      finishGame(true);
    }
  };

  // Finish the game
  const finishGame = (completed: boolean) => {
    setGameStarted(false);
    setGameOver(true);
    stopRecording();
    
    if (completed) {
      const completionBonus = difficultySettings[difficulty].completionBonus;
      setScore(prev => prev + completionBonus);
      setGameStats(prev => ({...prev, completionBonus}));
      toast.success(`Game completed! Bonus: +${completionBonus}`);
    }
    
    const gameSession = {
      id: gameSessionId,
      childId: 'current-child-id',
      gameType: 'sound-match',
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
    setCurrentRoundIndex(0);
    setSoundItems([]);
    setCurrentSound(null);
    setOptions([]);
    setScore(0);
    setTimeLeft(0);
    setTimeTaken(0);
    setIsCorrect(null);
    setGameOver(false);
    setSelectedOption(null);
    stopRecording();
  };

  // Exit the game
  const exitGame = () => {
    stopRecording();
    navigate('/games');
  };

  // Select difficulty
  const handleDifficultySelect = (level: 'beginner' | 'intermediate' | 'expert') => {
    setDifficulty(level);
  };

  // Replay the current sound
  const handleReplaySound = () => {
    if (currentSound) {
      playSound(currentSound);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bubblegum text-therapy-vibrant-purple mb-2">
          Sound Match
        </h1>
        <p className="text-lg">Listen to the sound and select the matching image!</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-3">
          <Card className="p-4">
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
                      <strong>Beginner:</strong> {difficultySettings.beginner.itemCount} sounds, {difficultySettings.beginner.optionsPerItem} options
                    </p>
                    <p className="mb-2">
                      <strong>Intermediate:</strong> {difficultySettings.intermediate.itemCount} sounds, {difficultySettings.intermediate.optionsPerItem} options
                    </p>
                    <p>
                      <strong>Expert:</strong> {difficultySettings.expert.itemCount} sounds, {difficultySettings.expert.optionsPerItem} options
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
                        <p className="font-medium">Correct Matches</p>
                        <p className="text-xl text-green-600">{gameStats.correct}</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg">
                        <p className="font-medium">Incorrect Matches</p>
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
                      <span className="font-medium">Sound:</span> {currentRoundIndex + 1}/{soundItems.length}
                    </div>
                  </div>
                  
                  <div className="flex flex-col items-center justify-center py-6">
                    <div className="mb-6 text-center">
                      <h2 className="text-lg text-gray-600 mb-2">Listen to the sound and select the matching item:</h2>
                      <Button 
                        onClick={handleReplaySound}
                        className="flex items-center gap-2 mb-6 bg-therapy-vibrant-blue"
                      >
                        <Volume2 className="h-5 w-5" />
                        {difficulty === 'beginner' ? 'Play Sound (Hint: ' + currentSound?.name + ')' : 'Play Sound'}
                      </Button>
                    </div>
                    
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 max-w-xl mx-auto">
                      {options.map((option, index) => (
                        <div 
                          key={index} 
                          onClick={() => selectedOption === null && checkAnswer(index)}
                          className={`
                            p-6 rounded-xl cursor-pointer transition-all text-center
                            ${selectedOption === index 
                              ? isCorrect 
                                ? 'bg-green-100 border-2 border-green-500' 
                                : 'bg-red-100 border-2 border-red-500'
                              : selectedOption !== null && option.id === currentSound.id
                                ? 'bg-green-100 border-2 border-green-500'
                                : 'bg-white border-2 border-gray-200 hover:bg-gray-50 hover:border-gray-300'
                            }
                          `}
                        >
                          <div className="text-5xl mb-2">{option.image}</div>
                          <div className="text-sm font-medium">{option.name}</div>
                        </div>
                      ))}
                    </div>
                    
                    {isCorrect === true && (
                      <div className="mt-6 text-green-600 font-bold text-lg animate-bounce">
                        Correct! Well done!
                      </div>
                    )}
                    
                    {isCorrect === false && (
                      <div className="mt-6 text-red-600 font-bold text-lg">
                        {difficulty === 'beginner' ? 'Try again!' : 'Wrong choice!'}
                      </div>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
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

export default SoundMatchGame;