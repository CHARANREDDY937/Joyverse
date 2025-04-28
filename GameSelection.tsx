
import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Card, CardContent } from '@/components/ui/card';
import { GameMetadata } from '@/types';
import { useAuth } from '@/context/AuthContext';
import {
  Brain,
  Puzzle,
  MessageSquareText,
  Music,
  Sparkles,
  VolumeX,
  Pencil,
  BookOpen
} from 'lucide-react';
import { toast } from 'sonner';

const GameSelection: React.FC = () => {
  const [difficulty, setDifficulty] = useState<'beginner' | 'intermediate' | 'expert'>('beginner');
  const { user, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  
  useEffect(() => {
    if (!isAuthenticated || !user || user.role !== 'child') {
      toast.error('You need to be logged in as a child to access games');
      navigate('/login');
    }
  }, [isAuthenticated, user, navigate]);
  
  const games: GameMetadata[] = [
    {
      id: 'memory-match',
      name: 'Memory Match',
      description: 'Match pairs of emoji cards to test your memory',
      iconName: 'Puzzle',
      color: 'bg-therapy-purple',
      available: true
    },
    {
      id: 'word-scramble',
      name: 'Word Scramble',
      description: 'Unscramble the letters to form words',
      iconName: 'MessageSquareText',
      color: 'bg-therapy-blue',
      available: true
    },
    {
      id: 'sound-match',
      name: 'Sound Match',
      description: 'Match sounds with the right pictures',
      iconName: 'Music',
      color: 'bg-therapy-yellow',
      available: true
    },
    {
      id: 'rhyme-time',
      name: 'Rhyme Time',
      description: 'Find words that rhyme together',
      iconName: 'Brain',
      color: 'bg-therapy-peach',
      available: true
    },
    {
      id: 'spell-bee',
      name: 'Spell Bee',
      description: 'Listen to a word and spell it correctly',
      iconName: 'Pencil',
      color: 'bg-therapy-pink',
      available: true
    },
    {
      id: 'word-play',
      name: 'Word Play',
      description: 'Match words in the same category',
      iconName: 'BookOpen',
      color: 'bg-therapy-green',
      available: true
    }
  ];

  const getIcon = (iconName: string) => {
    switch (iconName) {
      case 'Puzzle':
        return <Puzzle className="h-12 w-12" />;
      case 'MessageSquareText':
        return <MessageSquareText className="h-12 w-12" />;
      case 'Music':
        return <Music className="h-12 w-12" />;
      case 'Brain':
        return <Brain className="h-12 w-12" />;
      case 'Sparkles':
        return <Sparkles className="h-12 w-12" />;
      case 'Pencil':
        return <Pencil className="h-12 w-12" />;
      case 'BookOpen':
        return <BookOpen className="h-12 w-12" />;
      default:
        return <Puzzle className="h-12 w-12" />;
    }
  };

  const handleDifficultyChange = (level: 'beginner' | 'intermediate' | 'expert') => {
    setDifficulty(level);
  };

  return (
    <div className="container mx-auto p-4">
      <div className="text-center mb-10 animate-fade-in">
        <h1 className="text-4xl font-bubblegum text-therapy-vibrant-purple mb-2">
          Choose a Game
        </h1>
        <p className="text-xl">Select a fun game to play and practice your emotion skills!</p>
      </div>
      
      {/* Removed the Choose Your Difficulty section here */}
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 mb-10">
        {games.map((game) => (
          <div key={game.id} className="col-span-1">
            {game.available ? (
              <Link to={`/games/${game.id}`} state={{ difficulty }}>
                <Card
                  className="game-card group h-full"
                  // Add background image only for Memory Match
                  style={
                    game.id === 'memory-match'
                      ? {
                          backgroundImage: "url('/images/memory-match-bg.jpg')",
                          backgroundSize: 'cover',
                          backgroundPosition: 'center',
                          position: 'relative'
                        }
                      : undefined
                  }
                >
                  <CardContent
                    className={`flex flex-col items-center justify-center p-6 h-full ${
                      game.id === 'memory-match' ? 'bg-white/80 rounded-2xl' : ''
                    }`}
                    // Add a semi-transparent overlay for readability
                  >
                    <div className={`${game.color} p-5 rounded-full mb-4 group-hover:scale-110 transition-transform`}>
                      {getIcon(game.iconName)}
                    </div>
                    <h3 className="text-2xl font-bubblegum mb-2">{game.name}</h3>
                    <p className="text-center text-gray-600">{game.description}</p>
                    <div className="mt-6 bg-gradient-to-r from-therapy-vibrant-purple to-therapy-vibrant-blue text-white py-2 px-6 rounded-lg shadow-md group-hover:shadow-lg transition-all transform group-hover:-translate-y-1">
                      Play Now
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ) : (
              <Card className="game-card h-full opacity-70">
                <CardContent className="flex flex-col items-center justify-center p-6 h-full">
                  <div className={`${game.color} bg-opacity-60 p-5 rounded-full mb-4 relative`}>
                    {getIcon(game.iconName)}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <VolumeX className="h-6 w-6 text-gray-700" />
                    </div>
                  </div>
                  <h3 className="text-2xl font-bubblegum mb-2">{game.name}</h3>
                  <p className="text-center text-gray-500">{game.description}</p>
                  
                  <div className="mt-6 bg-gray-300 text-gray-600 py-2 px-6 rounded-lg">
                    Coming Soon
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        ))}
      </div>

      <div className="bg-white rounded-3xl shadow p-6 mb-8">
        <h2 className="text-2xl font-bubblegum mb-3 text-center">Game Descriptions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Memory Match</h3>
            <p>Match pairs of emotion cards to test your visual memory. The game helps improve recognition of emotional expressions.</p>
          </div>
          
          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Word Scramble</h3>
            <p>Unscramble letters to form words related to emotions and communication. Great for vocabulary building.</p>
          </div>
          
          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Sound Match</h3>
            <p>Listen to sounds and match them with the correct images. Improves auditory processing skills.</p>
          </div>
          
          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Rhyme Time</h3>
            <p>Find words that rhyme together. Enhances phonological awareness and word recognition skills.</p>
          </div>

          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Spell Bee</h3>
            <p>Listen to words and spell them correctly. Improves spelling skills and word recognition.</p>
          </div>
          
          <div className="p-4 bg-therapy-gray rounded-xl">
            <h3 className="text-xl font-bold mb-2">Word Play</h3>
            <p>Sort words into categories based on their meaning or relationship. Enhances vocabulary and understanding of word relationships.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameSelection;
