import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useNavigate, useLocation } from 'react-router-dom';
import { Home, RefreshCw, Volume2, Clock, Award, Check } from 'lucide-react';
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid';
import { useAuth } from '@/context/AuthContext';

// Word lists by difficulty level
const wordsByLevel = {
  beginner: [
    { word: 'cat', hint: 'A small furry animal that says meow', context: 'The ____ chased the mouse.' },
    { word: 'dog', hint: 'A pet that barks', context: 'The ____ wagged its tail.' },
    { word: 'sun', hint: 'It gives us light during the day', context: 'The ____ shines brightly.' },
    { word: 'run', hint: 'To move quickly on foot', context: 'I like to ____ in the park.' },
    { word: 'hat', hint: 'You wear it on your head', context: 'She put on her ____ before going outside.' },
    { word: 'pen', hint: 'You write with it', context: 'I need a ____ to write my name.' },
    { word: 'cup', hint: 'You drink from it', context: 'The ____ is full of water.' }
  ],
  intermediate: [
    { word: 'happy', hint: 'Feeling joy', context: 'She was ____ when she received the gift.' },
    { word: 'climb', hint: 'To go up', context: 'They ____ to the top of the hill.' },
    { word: 'laugh', hint: 'Sound you make when something is funny', context: 'The joke made me ____.' },
    { word: 'smile', hint: 'Expression on your face when you re happy', context: 'Her ____ brightened the room.' },
    { word: 'brave', hint: 'Not afraid', context: 'The ____ firefighter saved the cat.' },
    { word: 'quiet', hint: 'Not making noise', context: 'Please be ____ in the library.' },
    { word: 'friend', hint: 'Someone you like to spend time with', context: 'My ____ came over to play.' },
    { word: 'dream', hint: 'What you see when you sleep', context: 'I had a good ____ last night.' },
    { word: 'plant', hint: 'A living thing that grows in soil', context: 'The ____ needs water to grow.' },
    { word: 'cloud', hint: 'White fluffy thing in the sky', context: 'The ____ covered the sun.' }
  ],
  expert: [
    { word: 'beautiful', hint: 'Very pretty or attractive', context: 'The sunset was ____.' },
    { word: 'adventure', hint: 'An exciting experience', context: 'We went on an ____ in the forest.' },
    { word: 'important', hint: 'Something that matters a lot', context: 'It is ____ to brush your teeth.' },
    { word: 'different', hint: 'Not the same', context: 'My shoes are ____ from yours.' },
    { word: 'remember', hint: 'To not forget', context: 'Can you ____ where we parked the car?' },
    { word: 'question', hint: 'Something you ask to get information', context: 'I have a ____ about homework.' },
    { word: 'surprise', hint: 'Something unexpected', context: 'The birthday party was a ____.' },
    { word: 'strength', hint: 'Power or ability', context: 'The athlete showed great ____.' },
    { word: 'together', hint: 'In the same place or at the same time', context: 'Lets work ____ on this project.' },
    { word: 'probably', hint: 'Likely to happen', context: 'It will ____ rain tomorrow.' },
    { word: 'knowledge', hint: 'Information that you know', context: 'He has a lot of ____ about science.' },
    { word: 'difficult', hint: 'Not easy', context: 'The math problem was ____.' }
  ]
};

// Settings for different difficulty levels
const difficultySettings = {
  beginner: {
    wordCount: 5,
    timePerWord: 60,
    pointsPerCorrect: 10,
    fastBonus: 5,
    fastTime: 30,
    completionBonus: 10,
    showHints: true,
    showContext: true,
    autoSuggest: true
  },
  intermediate: {
    wordCount: 8,
    timePerWord: 45,
    pointsPerCorrect: 20,
    fastBonus: 10,
    fastTime: 20,
    completionBonus: 20,
    penaltyPerWrong: 5,
    showHints: false,
    showContext: true,
    autoSuggest: false
  },
  expert: {
    wordCount: 10,
    timePerWord: 30,
    pointsPerCorrect: 30,
    fastBonus: 15,
    fastTime: 10,
    completionBonus: 30,
    penaltyPerWrong: 10,
    showHints: false,
    showContext: false,
    autoSuggest: false
  }
};

const SpellBeeGame: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>(
    location.state?.difficulty || 'beginner'
  );
  const [gameStarted, setGameStarted] = useState<boolean>(false);
  const [currentWordIndex, setCurrentWordIndex] = useState<number>(0);
  const [words, setWords] = useState<any[]>([]);
  const [userInput, setUserInput] = useState<string>('');
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
  const [wrongAttempts, setWrongAttempts] = useState<number>(0);
  const [questions, setQuestions] = useState< any []>([]);

  // Refs for media recording
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  // Ref for speech synthesis
  const speechSynthesisRef = useRef<SpeechSynthesis | null>(null);
  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Initialize speech synthesis
  useEffect(() => {
    speechSynthesisRef.current = window.speechSynthesis;
    return () => {
      if (speechSynthesisRef.current) {
        speechSynthesisRef.current.cancel();
      }
      // Ensure camera is stopped when component unmounts
      stopRecording();
    };
  }, []);

  // Select random words based on difficulty
  const selectRandomWords = useCallback(() => {
    const levelWords = wordsByLevel[difficulty];
    const count = difficultySettings[difficulty].wordCount;
    const shuffled = [...levelWords].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }, [difficulty]);

  // Start camera recording
  const startRecording = async () => {
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        stopRecording();
      }

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
      streamRef.current.getTracks().forEach(track => {
        track.stop();
      });
      streamRef.current = null;
    }
    
    mediaRecorderRef.current = null;
    recordedChunksRef.current = [];
  };

  // Play word sound using speech synthesis
  const playWordSound = (word: string) => {
    if (speechSynthesisRef.current) {
      speechSynthesisRef.current.cancel();
      
      const utterance = new SpeechSynthesisUtterance(word);
      utterance.rate = 0.8;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      
      const voices = speechSynthesisRef.current.getVoices();
      const englishVoice = voices.find(voice => 
        voice.lang.includes('en') && voice.name.includes('Female')
      );
      if (englishVoice) {
        utterance.voice = englishVoice;
      }
      
      speechUtteranceRef.current = utterance;
      speechSynthesisRef.current.speak(utterance);
      
      toast.info(`Playing sound for: "${word}"`);
    } else {
      toast.error('Speech synthesis not supported in your browser');
    }
  };

  // Initialize game
  const startGame = useCallback(() => {
    const selectedWords = selectRandomWords();
    setWords(selectedWords);
    setCurrentWordIndex(0);
    setScore(0);
    setTimeLeft(difficultySettings[difficulty].timePerWord);
    setTimeTaken(0);
    setGameStarted(true);
    setIsCorrect(null);
    setGameOver(false);
    setUserInput('');
    setWrongAttempts(0);
    setGameStats({
      correct: 0,
      incorrect: 0,
      timeBonus: 0,
      completionBonus: 0
    });
    setQuestions([]);
    
    startRecording();
    toast.success('Game started! Listen to the word and spell it correctly.');
    playWordSound(selectedWords[0].word);
  }, [difficulty, selectRandomWords]);

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
    
    const newQuestion = {
      questionId: currentWordIndex + 1,
      answeredCorrectly: false,
      timeTaken: difficultySettings[difficulty].timePerWord
    };
    
    setQuestions(prev => [...prev, newQuestion]);
    
    setTimeout(() => {
      moveToNextWord();
    }, 1500);
  };

  // Check the user's answer
  const checkAnswer = () => {
    const currentWord = words[currentWordIndex].word;
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
      
      const newQuestion = {
        questionId: currentWordIndex + 1,
        answeredCorrectly: true,
        timeTaken: difficultySettings[difficulty].timePerWord - timeLeft
      };
      
      setQuestions(prev => [...prev, newQuestion]);
      
      toast.success('Correct spelling!');
      
      setTimeout(() => {
        moveToNextWord();
      }, 1500);
    } else {
      setWrongAttempts(prev => prev + 1);
      
      if (difficulty !== 'beginner' && difficultySettings[difficulty].penaltyPerWrong) {
        setScore(prev => Math.max(0, prev - difficultySettings[difficulty].penaltyPerWrong));
        toast.error(`Wrong spelling! -${difficultySettings[difficulty].penaltyPerWrong} points`);
      } else {
        toast.error('Wrong spelling! Try again');
      }
      
      if (difficulty === 'beginner' && wrongAttempts >= 1) {
        toast.info(`Hint: The word is "${currentWord}"`);
      }
      
      if (difficulty !== 'beginner') {
        setGameStats(prev => ({...prev, incorrect: prev.incorrect + 1}));
        
        const newQuestion = {
          questionId: currentWordIndex + 1,
          answeredCorrectly: false,
          timeTaken: difficultySettings[difficulty].timePerWord - timeLeft
        };
        
        setQuestions(prev => [...prev, newQuestion]);
        
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
      setTimeLeft(difficultySettings[difficulty].timePerWord);
      setUserInput('');
      setIsCorrect(null);
      setWrongAttempts(0);
      playWordSound(words[nextIndex].word);
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
      childId: user?.id || 'guest',
      gameType: 'spell-bee',
      startTime: new Date().toISOString(),
      endTime: new Date().toISOString(),
      score: score,
      difficulty: difficulty,
      questions: questions
    };
    
    console.log('Game session data:', gameSession);
  };

  // Reset the game
  const resetGame = () => {
    setDifficulty('beginner');
    setGameStarted(false);
    setCurrentWordIndex(0);
    setWords([]);
    setUserInput('');
    setScore(0);
    setTimeLeft(0);
    setTimeTaken(0);
    setIsCorrect(null);
    setGameOver(false);
    setWrongAttempts(0);
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

  // Replay the current word sound
  const handleReplaySound = () => {
    if (words[currentWordIndex]) {
      playWordSound(words[currentWordIndex].word);
    }
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
          Spell Bee
        </h1>
        <p className="text-lg">Listen to the word and spell it correctly!</p>
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
                      <strong>Beginner:</strong> {difficultySettings.beginner.wordCount} words, hints available
                    </p>
                    <p className="mb-2">
                      <strong>Intermediate:</strong> {difficultySettings.intermediate.wordCount} words, sentence context
                    </p>
                    <p>
                      <strong>Expert:</strong> {difficultySettings.expert.wordCount} words, no hints
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
                        <p className="font-medium">Correct Words</p>
                        <p className="text-xl text-green-600">{gameStats.correct}</p>
                      </div>
                      <div className="bg-white p-3 rounded-lg">
                        <p className="font-medium">Incorrect Words</p>
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
                    <div className="mb-6 text-center">
                      <h2 className="text-xl text-gray-600 mb-3">Listen to the word and spell it:</h2>
                      <Button 
                        onClick={handleReplaySound}
                        className="flex items-center gap-2 mb-4 bg-therapy-vibrant-blue"
                      >
                        <Volume2 className="h-5 w-5" />
                        Play Word
                      </Button>
                      
                      {(difficulty === 'beginner' || difficulty === 'intermediate') && words[currentWordIndex]?.context && (
                        <div className="bg-therapy-gray p-4 rounded-xl mb-4">
                          <p>{words[currentWordIndex].context}</p>
                        </div>
                      )}
                      
                      {difficulty === 'beginner' && words[currentWordIndex]?.hint && (
                        <div className="text-sm text-gray-500 mb-2">
                          <strong>Hint:</strong> {words[currentWordIndex].hint}
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
                        placeholder="Type the word here..."
                        autoComplete={difficultySettings[difficulty].autoSuggest ? 'on' : 'off'}
                        autoFocus
                      />
                    </div>
                    
                    <Button 
                      onClick={checkAnswer} 
                      disabled={!userInput.trim()} 
                      className="flex items-center gap-2 px-8 py-6 text-xl"
                    >
                      <Check className="h-5 w-5" /> Submit Answer
                    </Button>
                    
                    {isCorrect === true && (
                      <div className="mt-4 text-green-600 font-bold text-lg animate-bounce">
                        Correct! That's the right spelling!
                      </div>
                    )}
                    
                    {isCorrect === false && (
                      <div className="mt-4 text-red-600 font-bold text-lg">
                        {difficulty === 'beginner' 
                          ? 'Try again! Check your spelling.' 
                          : `The correct spelling is "${words[currentWordIndex]?.word}"`}
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

export default SpellBeeGame;