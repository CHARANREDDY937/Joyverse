import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import * as Tone from 'tone';
import './MusicMaker.css';

const INSTRUMENTS = {
  piano: new Tone.Synth().toDestination(),
  xylophone: new Tone.MetalSynth().toDestination(),
  drums: new Tone.MembraneSynth().toDestination(),
};

const NOTES = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'];
const COLORS = ['#FF0000', '#FFA500', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#8F00FF', '#FF69B4'];

const DRUM_PATTERNS = {
  rock: ['C2', null, 'C2', null, 'G2', null, 'C2', null],
  pop: ['C2', 'G2', 'C2', 'G2', 'C2', 'G2', 'C2', 'G2'],
  jazz: ['C2', null, 'G2', 'C2', null, 'G2', 'C2', null],
};

const MusicMaker = () => {
  const navigate = useNavigate();
  const [selectedInstrument, setSelectedInstrument] = useState('piano');
  const [isPlaying, setIsPlaying] = useState(false);
  const [recording, setRecording] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [tempo, setTempo] = useState(120);
  const [drumPattern, setDrumPattern] = useState('rock');
  const [error, setError] = useState(null);
  const [toneStarted, setToneStarted] = useState(false);

  // Initialize game and clear emotions log
  useEffect(() => {
    const initializeGame = async () => {
      // Clear previous emotions log on backend
      try {
        const response = await fetch('http://localhost:8000/clear_emotions_log', {
          method: 'POST',
        });
        const data = await response.json();
        if (data.status !== 'success') {
          setError('Failed to clear emotions log: ' + data.message);
        } else {
          setError(null);
        }
      } catch (error) {
        setError('Error clearing emotions log: ' + error.message);
      }
    };

    initializeGame();

    return () => {
      Tone.Transport.stop();
    };
  }, []);

  // Start Tone.js on user interaction
  const startTone = async () => {
    if (!toneStarted) {
      await Tone.start();
      setToneStarted(true);
    }
  };

  const playNote = (note) => {
    if (!toneStarted) return; // Prevent playing before Tone.js is started
    INSTRUMENTS[selectedInstrument].triggerAttackRelease(note, '8n');
    if (isRecording) {
      setRecording(prev => [...prev, { note, instrument: selectedInstrument, time: Tone.now() }]);
    }
  };

  const startRecording = () => {
    if (!toneStarted) return;
    setIsRecording(true);
    setRecording([]);
  };

  const stopRecording = () => {
    setIsRecording(false);
  };

  const playRecording = () => {
    if (!toneStarted || recording.length === 0) return;
    
    const now = Tone.now();
    recording.forEach(({ note, instrument, time }) => {
      const adjustedTime = now + (time - recording[0].time);
      INSTRUMENTS[instrument].triggerAttackRelease(note, '8n', adjustedTime);
    });
  };

  const toggleDrumBeat = () => {
    if (!toneStarted) return;
    if (isPlaying) {
      Tone.Transport.stop();
      setIsPlaying(false);
    } else {
      const pattern = new Tone.Sequence(
        (time, note) => {
          if (note) INSTRUMENTS.drums.triggerAttackRelease(note, '8n', time);
        },
        DRUM_PATTERNS[drumPattern],
        '8n'
      );
      
      pattern.start(0);
      Tone.Transport.bpm.value = tempo;
      Tone.Transport.start();
      setIsPlaying(true);
    }
  };

  const resetGame = async () => {
    try {
      const response = await fetch('http://localhost:8000/clear_emotions_log', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status !== 'success') {
        setError('Failed to clear emotions log: ' + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError('Error clearing emotions log: ' + error.message);
    }

    setRecording([]);
    setIsRecording(false);
    setIsPlaying(false);
    Tone.Transport.stop();
    setSelectedInstrument('piano');
    setDrumPattern('rock');
    setTempo(120);
  };

  return (
    <div className="music-maker" onClick={startTone}>
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="music-header">
        <h1>Music Maker</h1>
        <motion.button
          className="reset-btn"
          onClick={resetGame}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Reset
        </motion.button>
      </div>

      {error && (
        <motion.p
          className="error-message"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {error}
        </motion.p>
      )}

      <div className="music-workspace">
        <div className="controls">
          <div className="instrument-selector">
            <h3>Choose Instrument</h3>
            <div className="instrument-buttons">
              {Object.keys(INSTRUMENTS).map((instrument) => (
                <motion.button
                  key={instrument}
                  className={`instrument-btn ${selectedInstrument === instrument ? 'active' : ''}`}
                  onClick={() => setSelectedInstrument(instrument)}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {instrument.charAt(0).toUpperCase() + instrument.slice(1)}
                </motion.button>
              ))}
            </div>
          </div>

          <div className="drum-controls">
            <h3>Drum Beat</h3>
            <div className="pattern-selector">
              {Object.keys(DRUM_PATTERNS).map((pattern) => (
                <motion.button
                  key={pattern}
                  className={`pattern-btn ${drumPattern === pattern ? 'active' : ''}`}
                  onClick={() => setDrumPattern(pattern)}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {pattern.charAt(0).toUpperCase() + pattern.slice(1)}
                </motion.button>
              ))}
            </div>
            <div className="tempo-control">
              <label>Tempo: {tempo} BPM</label>
              <input
                type="range"
                min="60"
                max="180"
                value={tempo}
                onChange={(e) => setTempo(Number(e.target.value))}
              />
            </div>
            <motion.button
              className={`play-btn ${isPlaying ? 'active' : ''}`}
              onClick={toggleDrumBeat}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {isPlaying ? 'Stop Beat' : 'Play Beat'}
            </motion.button>
          </div>

          <div className="recording-controls">
            <h3>Recording</h3>
            <div className="record-buttons">
              <motion.button
                className={`record-btn ${isRecording ? 'active' : ''}`}
                onClick={isRecording ? stopRecording : startRecording}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </motion.button>
              <motion.button
                className="play-recording-btn"
                onClick={playRecording}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={recording.length === 0}
              >
                Play Recording
              </motion.button>
            </div>
          </div>
        </div>

        <div className="piano-keys">
          {NOTES.map((note, index) => (
            <motion.div
              key={note}
              className="key"
              style={{ backgroundColor: COLORS[index] }}
              onClick={() => playNote(note)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {note.charAt(0)}
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MusicMaker;