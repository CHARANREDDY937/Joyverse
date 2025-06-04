import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './WordWizard.css';
import { useNavigate } from 'react-router-dom';

const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const instructionBackground = 'https://i.pinimg.com/736x/e7/eb/1e/e7eb1e0579cc962ca5dfec740ae68f5e.jpg';

const WORD_CATEGORIES = {
  easy: {
    animals: ['CAT', 'DOG', 'COW', 'PIG', 'RAT', 'HEN', 'FOX', 'BAT', 'ANT', 'BEE', 'FISH', 'FROG', 'GOAT', 'LION', 'MOLE', 'MOTH', 'MULE', 'SEAL', 'SWAN', 'TOAD', 'WOLF'],
    fruits: ['APPLE', 'PEAR', 'GRAPE', 'PLUM', 'FIG', 'DATE', 'LIME', 'KIWI', 'PEACH', 'MANGO', 'GUAVA', 'LEMON', 'MELON', 'BERRY', 'PRUNE', 'CHERRY', 'PAPAYA', 'BANANA', 'ORANGE', 'APRICOT', 'QUINCE'],
    colors: ['RED', 'BLUE', 'GREEN', 'PINK', 'BLACK', 'WHITE', 'GREY', 'BROWN', 'GOLD', 'CORAL', 'BEIGE', 'IVORY', 'MAUVE', 'OLIVE', 'PEACH', 'PLUM', 'RUBY', 'SAND', 'TEAL', 'TAN', 'WINE']
  },
  medium: {
    animals: ['TIGER', 'ZEBRA', 'CAMEL', 'HORSE', 'SHEEP', 'KOALA', 'PANDA', 'EAGLE', 'SHARK', 'SNAKE', 'MOUSE', 'MOOSE', 'LLAMA', 'BISON', 'HIPPO', 'HYENA', 'LEMUR', 'OTTER', 'SLOTH', 'WHALE', 'RHINO'],
    fruits: ['DURIAN', 'RAISIN', 'LONGAN', 'LYCHEE', 'MULBERRY', 'NECTARINE', 'PASSION', 'PAWPAW', 'POMELO', 'SAPOTA', 'SOURSOP', 'TANGELO', 'CUSTARD', 'DAMSON', 'FEIJOA', 'KUMQUAT', 'MANGOSTEEN', 'PLANTAIN', 'QUANDONG', 'TAMARIND', 'UGLI'],
    colors: ['PURPLE', 'YELLOW', 'SILVER', 'AMBER', 'AZURE', 'BRONZE', 'COPPER', 'CREAM', 'CRIMSON', 'FUCHSIA', 'INDIGO', 'KHAKI', 'LEMON', 'LILAC', 'MAGENTA', 'MAROON', 'ORANGE', 'SALMON', 'SIENNA', 'VIOLET', 'WHEAT']
  },
  hard: {
    animals: ['GIRAFFE', 'ELEPHANT', 'KANGAROO', 'DOLPHIN', 'PENGUIN', 'OCTOPUS', 'LEOPARD', 'PLATYPUS', 'ANTELOPE', 'BADGER', 'CHEETAH', 'GAZELLE', 'HAMSTER', 'JAGUAR', 'MEERKAT', 'OCELOT', 'PANTHER', 'RACCOON', 'SQUIRREL', 'WALRUS', 'WOMBAT'],
    fruits: ['DRAGONFRUIT', 'PINEAPPLE', 'BLUEBERRY', 'JACKFRUIT', 'COCONUT', 'AVOCADO', 'BLACKBERRY', 'BOYSENBERRY', 'CANTALOUPE', 'CLEMENTINE', 'GOOSEBERRY', 'GRAPEFRUIT', 'HONEYDEW', 'MANDARIN', 'MULBERRY', 'PERSIMMON', 'POMEGRANATE', 'RASPBERRY', 'STRAWBERRY', 'TANGERINE', 'WATERMELON'],
    colors: ['TURQUOISE', 'BURGUNDY', 'LAVENDER', 'AMETHYST', 'AQUAMARINE', 'CERULEAN', 'CHARTREUSE', 'CINNAMON', 'EMERALD', 'MAHOGANY', 'MIDNIGHT', 'MUSTARD', 'PERIWINKLE', 'PLATINUM', 'PUMPKIN', 'SAPPHIRE', 'SCARLET', 'SEAGREEN', 'THISTLE', 'ULTRAMARINE', 'VERMILION']
  }
};

const WORD_HINTS = {
  easy: {
    animals: {
      'CAT': 'A furry pet that says meow',
      'DOG': 'Man\'s best friend that barks',
      'COW': 'Gives us milk and says moo',
      'PIG': 'Pink farm animal that says oink',
      'RAT': 'Small rodent with a long tail',
      'HEN': 'Female chicken that lays eggs',
      'FOX': 'Cunning orange wild animal',
      'BAT': 'Flying mammal active at night',
      'ANT': 'Tiny insect that works in colonies',
      'BEE': 'Makes honey and can sting',
      'FISH': 'Lives in water and has fins',
      'FROG': 'Green amphibian that jumps',
      'GOAT': 'Has horns and eats anything',
      'LION': 'King of the jungle',
      'MOLE': 'Lives underground and digs',
      'MOTH': 'Like a butterfly but nocturnal',
      'MULE': 'Mix between a horse and donkey',
      'SEAL': 'Swimming mammal that barks',
      'SWAN': 'Elegant white water bird',
      'TOAD': 'Like a frog but lives on land',
      'WOLF': 'Wild cousin of dogs'
    },
    fruits: {
      'APPLE': 'Red or green, keeps the doctor away',
      'PEAR': 'Shaped like a teardrop',
      'GRAPE': 'Small round fruit in clusters',
      'PLUM': 'Purple fruit with soft flesh',
      'FIG': 'Sweet fruit with many seeds inside',
      'DATE': 'Sweet brown fruit from palm trees',
      'LIME': 'Small green citrus fruit',
      'KIWI': 'Brown fuzzy outside, green inside',
      'PEACH': 'Soft fuzzy fruit with large pit',
      'MANGO': 'Sweet tropical orange fruit',
      'GUAVA': 'Tropical fruit with pink flesh',
      'LEMON': 'Sour yellow citrus fruit',
      'MELON': 'Large round sweet fruit',
      'BERRY': 'Small sweet fruit that grows on bushes',
      'PRUNE': 'Dried plum',
      'CHERRY': 'Small red fruit with a pit',
      'PAPAYA': 'Tropical fruit with black seeds',
      'BANANA': 'Yellow curved fruit',
      'ORANGE': 'Round citrus fruit',
      'APRICOT': 'Small orange fruit like a peach',
      'QUINCE': 'Yellow fruit used in jams'
    },
    colors: {
      'RED': 'Color of fire trucks',
      'BLUE': 'Color of the sky',
      'GREEN': 'Color of grass',
      'PINK': 'Light red color',
      'BLACK': 'Color of night',
      'WHITE': 'Color of snow',
      'GREY': 'Between black and white',
      'BROWN': 'Color of chocolate',
      'GOLD': 'Color of treasure',
      'CORAL': 'Pinkish-orange like sea creatures',
      'BEIGE': 'Light brown color',
      'IVORY': 'Off-white like elephant tusks',
      'MAUVE': 'Soft purple color',
      'OLIVE': 'Dark yellowish-green',
      'PEACH': 'Soft pink-orange',
      'PLUM': 'Dark purple like the fruit',
      'RUBY': 'Deep red like the gemstone',
      'SAND': 'Color of the beach',
      'TEAL': 'Blue-green like the ocean',
      'TAN': 'Light brown like leather',
      'WINE': 'Deep red like the drink'
    }
  },
  medium: {
    animals: {
      'TIGER': 'Big cat with orange and black stripes',
      'ZEBRA': 'Black and white striped horse',
      'CAMEL': 'Desert animal with humps',
      'HORSE': 'Animal you can ride',
      'SHEEP': 'Gives us wool',
      'KOALA': 'Australian animal that eats eucalyptus',
      'PANDA': 'Black and white bear from China',
      'EAGLE': 'Large bird of prey',
      'SHARK': 'Large fish with sharp teeth',
      'SNAKE': 'Reptile that slithers',
      'MOUSE': 'Small rodent that squeaks',
      'MOOSE': 'Large deer with big antlers',
      'LLAMA': 'South American animal like a camel',
      'BISON': 'Large wild cow',
      'HIPPO': 'Large water animal',
      'HYENA': 'Laughing wild dog',
      'LEMUR': 'Primate with a long striped tail',
      'OTTER': 'Playful water animal',
      'SLOTH': 'Very slow moving animal',
      'WHALE': 'Largest sea mammal',
      'RHINO': 'Large animal with horns on its nose'
    },
    fruits: {
      'DURIAN': 'Spiky fruit with strong smell',
      'RAISIN': 'Dried grape',
      'LONGAN': 'Small round Asian fruit',
      'LYCHEE': 'Sweet white fruit with red shell',
      'MULBERRY': 'Berry that grows on trees',
      'NECTARINE': 'Like a peach without fuzz',
      'PASSION': 'Purple fruit with seeds inside',
      'PAWPAW': 'Yellow tropical fruit',
      'POMELO': 'Large citrus fruit',
      'SAPOTA': 'Brown fruit with sweet flesh',
      'SOURSOP': 'Spiky green fruit',
      'TANGELO': 'Mix of tangerine and pomelo',
      'CUSTARD': 'Creamy yellow fruit',
      'DAMSON': 'Small purple plum',
      'FEIJOA': 'Green egg-shaped fruit',
      'KUMQUAT': 'Small orange citrus',
      'MANGOSTEEN': 'Purple fruit with white segments',
      'PLANTAIN': 'Cooking banana',
      'QUANDONG': 'Australian peach',
      'TAMARIND': 'Brown pod with sour pulp',
      'UGLI': 'Wrinkled citrus fruit'
    },
    colors: {
      'PURPLE': 'Color of grapes',
      'YELLOW': 'Color of the sun',
      'SILVER': 'Metallic gray color',
      'AMBER': 'Golden brown like tree sap',
      'AZURE': 'Bright sky blue',
      'BRONZE': 'Metallic brown',
      'COPPER': 'Metallic orange-brown',
      'CREAM': 'Off-white like milk',
      'CRIMSON': 'Deep bright red',
      'FUCHSIA': 'Bright pink-purple',
      'INDIGO': 'Deep blue-purple',
      'KHAKI': 'Light brown like army clothes',
      'LEMON': 'Bright yellow like the fruit',
      'LILAC': 'Light purple like the flower',
      'MAGENTA': 'Bright pink-purple',
      'MAROON': 'Dark brownish red',
      'ORANGE': 'Color of carrots',
      'SALMON': 'Pink-orange like the fish',
      'SIENNA': 'Reddish brown',
      'VIOLET': 'Purple like the flower',
      'WHEAT': 'Light tan like grain'
    }
  },
  hard: {
    animals: {
      'GIRAFFE': 'Tall animal with long neck',
      'ELEPHANT': 'Large gray animal with trunk',
      'KANGAROO': 'Hopping Australian animal',
      'DOLPHIN': 'Smart sea mammal',
      'PENGUIN': 'Black and white bird that swims',
      'OCTOPUS': 'Sea animal with eight arms',
      'LEOPARD': 'Big cat with spots',
      'PLATYPUS': 'Duck-billed mammal that lays eggs',
      'ANTELOPE': 'Fast running African animal',
      'BADGER': 'Black and white burrowing animal',
      'CHEETAH': 'Fastest land animal',
      'GAZELLE': 'Graceful jumping deer',
      'HAMSTER': 'Small pet rodent',
      'JAGUAR': 'Spotted big cat from Americas',
      'MEERKAT': 'Small animal that stands up',
      'OCELOT': 'Small spotted wild cat',
      'PANTHER': 'Large black cat',
      'RACCOON': 'Masked night animal',
      'SQUIRREL': 'Climbs trees and collects nuts',
      'WALRUS': 'Arctic animal with tusks',
      'WOMBAT': 'Australian burrowing animal'
    },
    fruits: {
      'DRAGONFRUIT': 'Pink scaly fruit with white flesh',
      'PINEAPPLE': 'Spiky tropical fruit',
      'BLUEBERRY': 'Small round blue fruit',
      'JACKFRUIT': 'Large spiky tropical fruit',
      'COCONUT': 'Hard brown fruit with water inside',
      'AVOCADO': 'Green fruit with large pit',
      'BLACKBERRY': 'Dark berry that grows on bushes',
      'BOYSENBERRY': 'Cross between several berries',
      'CANTALOUPE': 'Orange melon with netted skin',
      'CLEMENTINE': 'Small sweet orange',
      'GOOSEBERRY': 'Small tart green berry',
      'GRAPEFRUIT': 'Large pink citrus fruit',
      'HONEYDEW': 'Green sweet melon',
      'MANDARIN': 'Small orange citrus',
      'MULBERRY': 'Berry that grows on trees',
      'PERSIMMON': 'Orange fruit that\'s very sweet',
      'POMEGRANATE': 'Red fruit full of seeds',
      'RASPBERRY': 'Red bumpy berry',
      'STRAWBERRY': 'Red heart-shaped berry',
      'TANGERINE': 'Sweet orange citrus',
      'WATERMELON': 'Large green fruit with red inside'
    },
    colors: {
      'TURQUOISE': 'Blue-green like tropical water',
      'BURGUNDY': 'Dark red like wine',
      'LAVENDER': 'Light purple like the flower',
      'AMETHYST': 'Purple like the gemstone',
      'AQUAMARINE': 'Light blue-green like the ocean',
      'CERULEAN': 'Deep sky blue',
      'CHARTREUSE': 'Yellow-green color',
      'CINNAMON': 'Brown like the spice',
      'EMERALD': 'Deep green like the gemstone',
      'MAHOGANY': 'Dark reddish-brown wood',
      'MIDNIGHT': 'Very dark blue',
      'MUSTARD': 'Yellow-brown like the condiment',
      'PERIWINKLE': 'Light blue-violet',
      'PLATINUM': 'Silvery white metal',
      'PUMPKIN': 'Orange like Halloween',
      'SAPPHIRE': 'Deep blue like the gemstone',
      'SCARLET': 'Bright vivid red',
      'SEAGREEN': 'Dark blue-green',
      'THISTLE': 'Light purple-pink',
      'ULTRAMARINE': 'Deep ocean blue',
      'VERMILION': 'Bright orange-red'
    }
  }
};

const getLevelFromEmotion = (emotion) => {
  switch (emotion) {
    case 'Surprise':
    case 'Sadness':
    return 'easy';
    case 'Neutral':
    case 'Fear':
    return 'medium';
    case 'Anger':
    case 'Happiness':
    case 'Disgust':
    return 'hard';
    default:
      return 'medium'; // Fallback to medium if emotion is unrecognized
  }
};

const WordWizard = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [word, setWord] = useState('');
  const [guessedLetters, setGuessedLetters] = useState(new Set());
  const [remainingLives, setRemainingLives] = useState(6);
  const [gameStatus, setGameStatus] = useState('playing');
  const [score, setScore] = useState(0);
  const [category, setCategory] = useState('animals');
  const [error, setError] = useState(null);
  const [showInstructions, setShowInstructions] = useState(true);
  const [currentWordIndex, setCurrentWordIndex] = useState(0);
  const [currentSet, setCurrentSet] = useState(1);
  const [wordsCompleted, setWordsCompleted] = useState(0);
  const [currentLevel, setCurrentLevel] = useState('easy');
  const [pendingLevel, setPendingLevel] = useState(getLevelFromEmotion(emotion));

  const categories = useMemo(() => WORD_CATEGORIES[currentLevel], [currentLevel]);

  // Update pendingLevel when emotion changes
  useEffect(() => {
    setPendingLevel(getLevelFromEmotion(emotion));
  }, [emotion]);

  const selectNewWord = useCallback(() => {
    const words = categories[category];
    const startIndex = ((currentSet - 1) * 3) % words.length;
    const currentIndex = (startIndex + currentWordIndex) % words.length;
    const newWord = words[currentIndex];
    setWord(newWord);
    setGuessedLetters(new Set());
    setRemainingLives(6);
    setGameStatus('playing');
  }, [category, categories, currentWordIndex, currentSet]);

  useEffect(() => {
    selectNewWord();
  }, [selectNewWord]);

  const handleLetterGuess = (letter) => {
    if (gameStatus !== 'playing') return;
    const newGuessedLetters = new Set(guessedLetters);
    newGuessedLetters.add(letter);
    setGuessedLetters(newGuessedLetters);

    if (!word.includes(letter)) {
      setRemainingLives((lives) => lives - 1);
    }

    checkGameStatus(newGuessedLetters);
  };

  const handleGameWin = () => {
    setGameStatus('won');
    setScore((score) => score + 100);
    setWordsCompleted((prev) => prev + 1);

    // Apply pending level after winning
    setCurrentLevel(pendingLevel);

    if (wordsCompleted + 1 >= 3) {
      setCurrentSet((prev) => prev + 1);
      setCurrentWordIndex(0);
      setWordsCompleted(0);
    } else {
      setCurrentWordIndex((prev) => prev + 1);
    }
  };

  const checkGameStatus = (newGuessedLetters) => {
    if (word.split('').every((letter) => newGuessedLetters.has(letter))) {
      handleGameWin();
    } else if (remainingLives <= 1) {
      setGameStatus('lost');
      // Apply pending level after losing
      setCurrentLevel(pendingLevel);
    }
  };

  const resetGame = async () => {
    // try {
    //   const response = await fetch('http://localhost:8000/clear_emotions_log', {
    //     method: 'POST',
    //   });
    //   const data = await response.json();
    //   if (data.status !== 'success') {
    //     setError('Failed to clear emotions log: ' + data.message);
    //   } else {
    //     setError(null);
    //   }
    // } catch (error) {
    //   setError('Error clearing emotions log: ' + error.message);
    // }

    setScore(0);
    setCategory('animals');
    setWordsCompleted(0);
    setCurrentWordIndex(0);
    setCurrentSet(1);
    setGuessedLetters(new Set());
    setRemainingLives(6);
    setGameStatus('playing');
    setCurrentLevel(pendingLevel);
    selectNewWord();
  };

  const handleBack = () => navigate('/child/games');

  const backgroundImage = showInstructions ? instructionBackground : (emotionBackgrounds[emotion] || emotionBackgrounds.Neutral);

  const renderInstructions = () => (
    <motion.div 
      className="instructions" 
      initial={{ opacity: 0, y: 20 }} 
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
    >
      <h1>Welcome to Word Wizard!</h1>
      <p>🎯 Guess the hidden word letter by letter.</p>
      <p>❤ You have 6 lives. Each wrong guess costs a life.</p>
      <p>🏆 Complete 3 words to progress to the next set!</p>
      <p>📚 The game adjusts based on your emotions after each word.</p>
      <motion.button
        className="start-btn"
        onClick={() => setShowInstructions(false)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        Start Game
      </motion.button>
    </motion.div>
  );

  const renderGame = () => (
    <div className="game-section">
      <motion.button
        className="back-button"
        onClick={handleBack}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header-container">
        <h1 className="game-title">Word Wizard</h1>
        <div className="level-status">
          Level: {currentLevel.toUpperCase()} ({wordsCompleted}/3)
          <div className="set-status">Set: {currentSet}</div>
        </div>
        <div className="score-lives">
          <span className="score-text">Score: {score}</span>
          <span className="lives-text">Lives: {'❤'.repeat(remainingLives)}</span>
        </div>
      </div>

      {error && (
        <motion.p 
          className="error-message" 
          initial={{ opacity: 0 }} 
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {error}
        </motion.p>
      )}

      <div className="category-selector">
        {Object.keys(categories).map((cat) => (
          <motion.button
            key={cat}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className={`category-btn ${category === cat ? 'active' : ''}`}
            onClick={() => {
              setCategory(cat);
              setCurrentWordIndex(0);
              setCurrentSet(1);
              setWordsCompleted(0);
            }}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </motion.button>
        ))}
      </div>

      <div className="word-display">
        {word.split('').map((letter, index) => (
          <motion.div
            key={index}
            className="letter-box"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.1 }}
          >
            {guessedLetters.has(letter) ? letter : '_'}
          </motion.div>
        ))}
      </div>

      <div className="hint-container">
        <motion.p 
          className="hint-text"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          Hint: {WORD_HINTS[currentLevel]?.[category]?.[word] || "No hint available"}
        </motion.p>
      </div>

      <div className="keyboard">
        {Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ').map((letter) => (
          <motion.button
            key={letter}
            className={`letter-btn ${guessedLetters.has(letter) ? 'used' : ''}`}
            onClick={() => handleLetterGuess(letter)}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            disabled={guessedLetters.has(letter) || gameStatus !== 'playing'}
          >
            {letter}
          </motion.button>
        ))}
      </div>

      {gameStatus !== 'playing' && (
        <motion.div 
          className="game-over" 
          initial={{ scale: 0 }} 
          animate={{ scale: 1 }}
          exit={{ scale: 0 }}
        >
          <h2>{gameStatus === 'won' ? '🎉 Congratulations!' : '😢 Game Over'}</h2>
          <p>The word was: {word}</p>
          <motion.button
            className="play-again-btn"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={selectNewWord}
          >
            Next Word
          </motion.button>
          <motion.button
            className="reset-btn"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={resetGame}
          >
            Start Over
          </motion.button>
        </motion.div>
      )}
    </div>
  );

  return (
    <div
      className="word-wizard"
      style={{
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        minHeight: '100vh',
        transition: 'background-image 0.5s ease-in-out',
      }}
    >
      <div className="content-wrapper">
        <AnimatePresence mode="wait">
          {showInstructions ? renderInstructions() : renderGame()}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default WordWizard;