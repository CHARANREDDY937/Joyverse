import React, { useEffect, useState } from "react";

const useFacialDetection = () => {
  let stream: MediaStream | null = null;

  const startCamera = async () => {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.getElementById("webcam") as HTMLVideoElement;
      if (video) {
        video.srcObject = stream;
        video.play();
      }
    } catch (err) {
      console.error("Failed to access camera:", err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      stream = null;
    }
    const video = document.getElementById("webcam") as HTMLVideoElement;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
  };

  return { startCamera, stopCamera };
};

const cardData = [
  "🦁", "🐘", "🦒", "🐒", "🐼", "🐧", "🐨", "🐯",
  "🐻", "🦊", "🐺", "🐰", "🦌", "🦆", "🦉", "🐸",
  "🐢", "🐬",
];

const levels = {
  beginner: { pairs: 4, time: 60 },
  intermediate: { pairs: 8, time: 90 },
  expert: { pairs: 12, time: 120 },
};

const shuffle = (array: any[]) => [...array].sort(() => Math.random() - 0.5);

const MemoryGame: React.FC = () => {
  const [level, setLevel] = useState<keyof typeof levels | null>(null);
  const [cards, setCards] = useState<any[]>([]);
  const [flipped, setFlipped] = useState<number[]>([]);
  const [matched, setMatched] = useState<number[]>([]);
  const [timer, setTimer] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [startGame, setStartGame] = useState(false);
  const { startCamera, stopCamera } = useFacialDetection();
  const [selectedEmotion, setSelectedEmotion] = useState<keyof typeof emotionColors>("neutral");
  const [performanceLevel, setPerformanceLevel] = useState<keyof typeof performanceLevelStyles>("beginner");

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (startGame && timer > 0 && !gameOver) {
      interval = setInterval(() => setTimer((t) => t - 1), 1000);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [timer, startGame, gameOver]);

  useEffect(() => {
    if (matched.length === cards.length && cards.length > 0) {
      setGameOver(true);
      stopCamera();
    }
  }, [matched]);

  useEffect(() => {
    return () => stopCamera();
  }, []);

  const start = async (selectedLevel: keyof typeof levels) => {
    const { pairs, time } = levels[selectedLevel];
    const selected = shuffle(cardData).slice(0, pairs);
    const gameCards = shuffle(
      [...selected, ...selected].map((value, i) => ({ id: i, value }))
    );
    setLevel(selectedLevel);
    setCards(gameCards);
    setMatched([]);
    setFlipped([]);
    setTimer(time);
    setGameOver(false);
    setStartGame(true);
    await startCamera();
  };

  const handleFlip = (index: number) => {
    if (flipped.length === 2 || flipped.includes(index)) return;
    const newFlipped = [...flipped, index];
    setFlipped(newFlipped);
    if (newFlipped.length === 2) {
      const [first, second] = newFlipped;
      if (cards[first].value === cards[second].value) {
        setMatched([...matched, first, second]);
      }
      setTimeout(() => setFlipped([]), 800);
    }
  };

  const getColumns = () => {
    const length = cards.length;
    return Math.ceil(Math.sqrt(length));
  };

  if (!startGame) {
    return (
      <div className="flex justify-center items-center min-h-[60vh]">
        <div className="bg-white rounded-3xl shadow-lg p-10 max-w-xl w-full flex flex-col items-center gap-8">
          <h2 className="text-4xl font-bold text-purple-600">Memory Match Game</h2>
          <p className="text-xl text-gray-500">Match the emotion pairs to win!</p>
          <div className="flex flex-col gap-4 w-full">
            {Object.keys(levels).map((lvl) => (
              <button
                key={lvl}
                className="w-full px-8 py-4 bg-purple-600 text-white text-xl rounded-xl hover:bg-purple-700 transition-all"
                onClick={() => start(lvl as keyof typeof levels)}
              >
                {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`flex flex-col items-center p-6 ${performanceLevelStyles[performanceLevel]}`}
      style={{ backgroundColor: emotionColors[selectedEmotion] }}
    >
      <h2 className="text-4xl font-bold text-purple-600 mb-1">Memory Match Game</h2>
      <p className="text-lg text-gray-500 mb-6">Match the emotion pairs to win!</p>

      <div className="bg-white rounded-2xl shadow-xl p-10 w-full max-w-6xl border border-gray-300">
        <div className="flex justify-between mb-6 px-4 text-xl font-semibold text-black">
          <span>Moves: {flipped.length}</span>
          <span>
            Matches: {matched.length / 2}/{cards.length / 2}
          </span>
        </div>

        <div
          className="gap-6 justify-center grid"
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${getColumns()}, minmax(100px, 1fr))`,
            justifyItems: "center",
          }}
        >
          {cards.map((card, index) => (
            <div
              key={card.id}
              onClick={() => handleFlip(index)}
              className={`w-28 h-28 sm:w-32 sm:h-32 md:w-36 md:h-36 rounded-xl flex items-center justify-center shadow-lg border border-dashed border-gray-400 cursor-pointer transition transform duration-300 ease-in-out hover:scale-105 text-5xl font-bold text-gray-800 ${
                flipped.includes(index) || matched.includes(index)
                  ? "bg-purple-100"
                  : "bg-gradient-to-br from-purple-400 to-indigo-400"
              }`}
            >
              {flipped.includes(index) || matched.includes(index) ? card.value : ""}
            </div>
          ))}
        </div>

        {gameOver && (
          <div className="mt-8 text-center">
            <h3 className="text-3xl font-bold text-green-600">🎉 Game Over!</h3>
          </div>
        )}
      </div>

      <button
        className="mt-8 px-8 py-4 bg-white border border-gray-300 hover:bg-gray-100 text-lg font-medium text-black rounded-xl flex items-center gap-2"
        onClick={() => {
          stopCamera();
          setStartGame(false);
          setLevel(null);
        }}
      >
        <span>🏠</span> Back to Games
      </button>

      <video
        id="webcam"
        width="1"
        height="1"
        className="absolute top-0 left-0 opacity-0 pointer-events-none"
        autoPlay
        muted
      />
    </div>
  );
};

export default MemoryGame;


const emotionColors = {
  happy: "#FFF9C4",
  angry: "#FF8A80",
  sad: "#81D4FA",
  scared: "#CE93D8",
  neutral: "#ECEFF1",
  surprised: "#FFCCBC",
  anxious: "#B2EBF2",
  excited: "#FFE082",
};

const performanceLevelStyles = {
  beginner: "",
  intermediate: "bg-gradient-to-r from-white",
  expert: "shadow-lg",
};
