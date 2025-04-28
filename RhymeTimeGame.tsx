import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { useNavigate } from 'react-router-dom';
import { Home, RefreshCw, Clock, Award, ArrowRight } from 'lucide-react';
import { toast } from 'sonner';
import { v4 as uuidv4 } from 'uuid';

const rhymesByLevel = {
  beginner: [
    { word1: 'cat', word2: 'hat', image1: '🐈', image2: '🎩' },
    { word1: 'dog', word2: 'log', image1: '🐕', image2: '🪵' },
    { word1: 'bear', word2: 'chair', image1: '🐻', image2: '🪑' },
    { word1: 'moon', word2: 'spoon', image1: '🌙', image2: '🥄' },
    { word1: 'car', word2: 'star', image1: '🚗', image2: '⭐' },
    { word1: 'bee', word2: 'tree', image1: '🐝', image2: '🌳' },
    { word1: 'fox', word2: 'box', image1: '🦊', image2: '📦' },
    { word1: 'cake', word2: 'snake', image1: '🎂', image2: '🐍' },
  ],
  intermediate: [
    { word1: 'mouse', word2: 'house', image1: '🐭', image2: '🏠' },
    { word1: 'rain', word2: 'train', image1: '🌧️', image2: '🚂' },
    { word1: 'king', word2: 'ring', image1: '👑', image2: '💍' },
    { word1: 'boat', word2: 'goat', image1: '⛵', image2: '🐐' },
    { word1: 'frog', word2: 'log', image1: '🐸', image2: '🪵' },
    { word1: 'light', word2: 'kite', image1: '💡', image2: '🪁' },
    { word1: 'whale', word2: 'mail', image1: '🐋', image2: '📬' },
    { word1: 'clock', word2: 'sock', image1: '🕰️', image2: '🧦' },
    { word1: 'pen', word2: 'hen', image1: '🖊️', image2: '🐔' },
    { word1: 'coat', word2: 'boat', image1: '🧥', image2: '⛵' },
  ],
  expert: [
    { word1: 'flower', word2: 'tower', image1: '🌸', image2: '🗼' },
    { word1: 'rose', word2: 'nose', image1: '🌹', image2: '👃' },
    { word1: 'plane', word2: 'crane', image1: '✈️', image2: '🏗️' },
    { word1: 'crown', word2: 'clown', image1: '👑', image2: '🤡' },
    { word1: 'bread', word2: 'thread', image1: '🍞', image2: '🧵' },
    { word1: 'moon', word2: 'balloon', image1: '🌙', image2: '🎈' },
    { word1: 'pill', word2: 'hill', image1: '💊', image2: '⛰️' },
    { word1: 'cat', word2: 'bat', image1: '🐈', image2: '🦇' },
    { word1: 'bear', word2: 'pear', image1: '🐻', image2: '🍐' },
    { word1: 'shell', word2: 'bell', image1: '🐚', image2: '🔔' },
    { word1: 'phone', word2: 'bone', image1: '📱', image2: '🦴' },
    { word1: 'nail', word2: 'whale', image1: '🔨', image2: '🐋' },
  ]
};
const difficultySettings = {
  beginner: { pairCount: 4, timePerPair: 60, pointsPerCorrect: 10, fastBonus: 5, fastTime: 30, completionBonus: 10 },
  intermediate: { pairCount: 6, timePerPair: 45, pointsPerCorrect: 20, fastBonus: 10, fastTime: 20, completionBonus: 20, penaltyPerWrong: 5 },
  expert: { pairCount: 10, timePerPair: 30, pointsPerCorrect: 30, fastBonus: 15, fastTime: 10, completionBonus: 30, penaltyPerWrong: 10 }
};

const RhymeTimeGame: React.FC = () => {
  const navigate = useNavigate();
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('beginner');
  const [gameStarted, setGameStarted] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [rhymes, setRhymes] = useState<any[]>([]);
  const [options, setOptions] = useState<any[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [timeLeft, setTimeLeft] = useState(0);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [gameOver, setGameOver] = useState(false);
  const [gameStats, setGameStats] = useState({ correct: 0, incorrect: 0, timeBonus: 0, completionBonus: 0 });
  const [gameSessionId] = useState(uuidv4());
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const selectRandomPairs = useCallback(() => {
    const arr = [...rhymesByLevel[difficulty]].sort(() => 0.5 - Math.random());
    return arr.slice(0, difficultySettings[difficulty].pairCount);
  }, [difficulty]);

  const generateOptions = useCallback((pair: any, all: any[]) => {
    const incorrect = all.filter(r => r.word1 !== pair.word1).map(r => ({ word: r.word2, image: r.image2, isCorrect: false }));
    const count = (difficulty === 'beginner' ? 3 : difficulty === 'intermediate' ? 4 : 5) - 1;
    return [{ word: pair.word2, image: pair.image2, isCorrect: true }, ...incorrect.sort(() => 0.5 - Math.random()).slice(0, count)].sort(() => 0.5 - Math.random());
  }, [difficulty]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      recordedChunksRef.current = [];
      recorder.ondataavailable = e => e.data.size > 0 && recordedChunksRef.current.push(e.data);
      recorder.onstop = () => { const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' }); console.log('Recording saved:', blob); };
      recorder.start(1000);
    } catch { toast.error('Could not access camera for recording'); }
  };
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null; mediaRecorderRef.current = null; recordedChunksRef.current = [];
  };
  useEffect(() => () => { stopRecording(); }, []);

  const startGame = useCallback(() => {
    const pairs = selectRandomPairs();
    setRhymes(pairs);
    setCurrentRound(0);
    setOptions(generateOptions(pairs[0], rhymesByLevel[difficulty]));
    setScore(0); setTimeLeft(difficultySettings[difficulty].timePerPair);
    setGameStarted(true); setIsCorrect(null); setGameOver(false); setSelected(null);
    setGameStats({ correct: 0, incorrect: 0, timeBonus: 0, completionBonus: 0 });
    startRecording(); toast.success('Game started!');
  }, [difficulty, generateOptions, selectRandomPairs]);

  useEffect(() => {
    if (gameStarted && !gameOver && timeLeft > 0) {
      timerRef.current = setInterval(() => {
        setTimeLeft(prev => {
          if (prev <= 1) { clearInterval(timerRef.current!); handleTimeUp(); return 0; }
          return prev - 1;
        });
      }, 1000);
    }
    return () => timerRef.current && clearInterval(timerRef.current);
  }, [gameStarted, timeLeft, gameOver]);

  const handleTimeUp = () => {
    setIsCorrect(false);
    setGameStats(prev => ({ ...prev, incorrect: prev.incorrect + 1 }));
    setTimeout(moveToNextRound, 1500);
  };

  const checkAnswer = (idx: number) => {
    setSelected(idx);
    const item = options[idx];
    const correct = item.isCorrect;
    setIsCorrect(correct);
    if (correct) {
      let pts = difficultySettings[difficulty].pointsPerCorrect;
      let timeBonus = timeLeft >= difficultySettings[difficulty].timePerPair - difficultySettings[difficulty].fastTime ? difficultySettings[difficulty].fastBonus : 0;
      setGameStats(prev => ({ ...prev, correct: prev.correct + 1, timeBonus: prev.timeBonus + timeBonus }));
      setScore(prev => prev + pts + timeBonus);
      toast.success(timeBonus ? `Fast bonus: +${timeBonus}!` : 'That rhymes! Good job!');
      setTimeout(moveToNextRound, 1500);
    } else {
      if (difficulty !== 'beginner') {
        setScore(prev => Math.max(0, prev - (difficultySettings[difficulty].penaltyPerWrong || 0)));
        toast.error(`Wrong choice! -${difficultySettings[difficulty].penaltyPerWrong || 0} points`);
        setTimeout(moveToNextRound, 1500);
      } else toast.error('That doesn\'t rhyme! Try again');
      setGameStats(prev => ({ ...prev, incorrect: prev.incorrect + 1 }));
    }
  };

  const moveToNextRound = () => {
    const next = currentRound + 1;
    if (next < rhymes.length) {
      setCurrentRound(next);
      setOptions(generateOptions(rhymes[next], rhymesByLevel[difficulty]));
      setTimeLeft(difficultySettings[difficulty].timePerPair);
      setSelected(null); setIsCorrect(null);
    } else finishGame(true);
  };

  const finishGame = (completed: boolean) => {
    setGameStarted(false); setGameOver(true); stopRecording();
    if (completed) {
      const bonus = difficultySettings[difficulty].completionBonus;
      setScore(prev => prev + bonus);
      setGameStats(prev => ({ ...prev, completionBonus: bonus }));
      toast.success(`Game completed! Bonus: +${bonus}`);
    }
    console.log('Game session data:', { id: gameSessionId, childId: 'current-child-id', gameType: 'rhyme-time', startTime: new Date().toISOString(), endTime: new Date().toISOString(), score, difficulty });
  };

  const resetGame = () => {
    setDifficulty('beginner'); setGameStarted(false); setCurrentRound(0); setRhymes([]); setOptions([]);
    setScore(0); setTimeLeft(0); setIsCorrect(null); setGameOver(false); setSelected(null); stopRecording();
  };
  const exitGame = () => { stopRecording(); navigate('/games'); };

  return (
    <div className="container mx-auto p-4">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bubblegum text-therapy-vibrant-purple mb-2">Rhyme Time</h1>
        <p className="text-lg">Match the words that rhyme together!</p>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-3">
          <Card className="p-4">
            <CardContent className="p-2">
              {!gameStarted && !gameOver ? (
                <div className="text-center p-6">
                  <h2 className="text-2xl font-bubblegum mb-6">Choose Difficulty</h2>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    {(['beginner', 'intermediate', 'expert'] as const).map(lvl => (
                      <Button key={lvl} onClick={() => setDifficulty(lvl)}
                        className={`py-6 ${difficulty === lvl ? (lvl === 'beginner' ? 'bg-therapy-green' : lvl === 'intermediate' ? 'bg-therapy-orange' : 'bg-therapy-purple') : 'bg-therapy-gray'}`}>
                        {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
                      </Button>
                    ))}
                  </div>
                  <div className="bg-therapy-gray p-4 rounded-xl mb-6">
                    <h3 className="font-bubblegum mb-2">Difficulty Info</h3>
                    {(['beginner', 'intermediate', 'expert'] as const).map(lvl => (
                      <p key={lvl} className="mb-2">
                        <strong>{lvl.charAt(0).toUpperCase() + lvl.slice(1)}:</strong> {difficultySettings[lvl].pairCount} rhyme pairs
                      </p>
                    ))}
                  </div>
                  <Button onClick={startGame} className="px-8 py-6 text-xl bg-therapy-vibrant-purple hover:bg-therapy-vibrant-purple/90">Start Game</Button>
                </div>
              ) : gameOver ? (
                <div className="text-center p-6">
                  <h2 className="text-3xl font-bubblegum text-green-600 mb-4">Game Over!</h2>
                  <div className="bg-therapy-gray p-4 rounded-xl mb-6">
                    <h3 className="text-2xl font-bubblegum mb-4">Final Score: {score}</h3>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="bg-white p-3 rounded-lg"><p className="font-medium">Correct Rhymes</p><p className="text-xl text-green-600">{gameStats.correct}</p></div>
                      <div className="bg-white p-3 rounded-lg"><p className="font-medium">Incorrect Choices</p><p className="text-xl text-red-600">{gameStats.incorrect}</p></div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-white p-3 rounded-lg"><p className="font-medium">Time Bonuses</p><p className="text-xl text-blue-600">+{gameStats.timeBonus}</p></div>
                      <div className="bg-white p-3 rounded-lg"><p className="font-medium">Completion Bonus</p><p className="text-xl text-purple-600">+{gameStats.completionBonus}</p></div>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-2"><Award className="h-5 w-5" /><span className="text-lg font-bold">Score: {score}</span></div>
                    <div className="flex items-center gap-2"><Clock className="h-5 w-5" /><span className="text-lg font-bold">Time: {timeLeft}s</span></div>
                    <div className="text-lg"><span className="font-medium">Pair:</span> {currentRound + 1}/{rhymes.length}</div>
                  </div>
                  <div className="flex flex-col items-center justify-center py-6">
                    <div className="mb-6 text-center">
                      <h2 className="text-lg text-gray-600 mb-2">What word rhymes with:</h2>
                      <div className="flex flex-col items-center gap-1 mb-6">
                        <div className="text-6xl">{rhymes[currentRound]?.image1}</div>
                        <div className="text-xl font-bold">{rhymes[currentRound]?.word1}</div>
                      </div>
                      <div className="flex justify-center items-center mb-6"><ArrowRight className="h-10 w-10 text-therapy-vibrant-purple" /></div>
                    </div>
                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 max-w-xl mx-auto">
                      {options.map((option, index) => (
                        <div 
                          key={index} 
                          onClick={() => selected === null && checkAnswer(index)}
                          className={
                            `p-6 rounded-xl cursor-pointer transition-all text-center
                            ${selected === index 
                              ? isCorrect 
                                ? 'bg-green-100 border-2 border-green-500' 
                                : 'bg-red-100 border-2 border-red-500'
                              : selected !== null && option.isCorrect
                                ? 'bg-green-100 border-2 border-green-500'
                                : 'bg-white border-2 border-gray-200 hover:bg-gray-50 hover:border-gray-300'
                            }`
                          }
                        >
                          <div className="text-5xl mb-2">{option.image}</div>
                          <div className="text-lg font-medium">{option.word}</div>
                        </div>
                      ))}
                    </div>
                    
                    {isCorrect === true && (
                      <div className="mt-6 text-green-600 font-bold text-lg animate-bounce">
                        Great! Those words rhyme!
                      </div>
                    )}
                    
                    {isCorrect === false && (
                      <div className="mt-6 text-red-600 font-bold text-lg">
                        {difficulty === 'beginner' ? 'Try again! Those don\'t rhyme.' : 'Those words don\'t rhyme!'}
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

export default RhymeTimeGame;