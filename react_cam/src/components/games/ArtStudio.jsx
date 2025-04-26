import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './ArtStudio.css';

const COLORS = [
  '#FF0000', '#FFA500', '#FFFF00', '#008000', 
  '#0000FF', '#4B0082', '#EE82EE', '#000000', 
  '#FFFFFF', '#FFC0CB', '#964B00', '#808080'
];

const BRUSH_SIZES = [2, 5, 10, 15, 20, 25];

const STICKERS = [
  '🌟', '🌈', '🌺', '🦋', '🐬', '🌙',
  '🌞', '🍀', '🎨', '🎭', '🦄', '🌸'
];

const ArtStudio = () => {
  const navigate = useNavigate();
  const canvasRef = useRef(null);
  const [ctx, setCtx] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [color, setColor] = useState('#000000');
  const [brushSize, setBrushSize] = useState(5);
  const [tool, setTool] = useState('brush'); // brush, eraser, sticker
  const [selectedSticker, setSelectedSticker] = useState('🌟');

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    const context = canvas.getContext('2d');
    context.lineCap = 'round';
    context.lineJoin = 'round';
    setCtx(context);
  }, []);

  const startDrawing = (e) => {
    const { offsetX, offsetY } = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(offsetX, offsetY);
    setIsDrawing(true);

    if (tool === 'sticker') {
      ctx.font = `${brushSize * 2}px Arial`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(selectedSticker, offsetX, offsetY);
    }
  };

  const draw = (e) => {
    if (!isDrawing || tool === 'sticker') return;
    const { offsetX, offsetY } = getCoordinates(e);
    
    ctx.lineWidth = brushSize;
    ctx.strokeStyle = tool === 'eraser' ? '#FFFFFF' : color;
    
    ctx.lineTo(offsetX, offsetY);
    ctx.stroke();
  };

  const stopDrawing = () => {
    ctx.closePath();
    setIsDrawing(false);
  };

  const getCoordinates = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX || e.touches[0].clientX;
    const y = e.clientY || e.touches[0].clientY;
    return {
      offsetX: x - rect.left,
      offsetY: y - rect.top
    };
  };

  const clearCanvas = () => {
    ctx.fillStyle = '#FFFFFF';
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  const saveDrawing = () => {
    const link = document.createElement('a');
    link.download = 'my-artwork.png';
    link.href = canvasRef.current.toDataURL();
    link.click();
  };

  return (
    <div className="art-studio">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="art-header">
        <h1>Art Studio</h1>
      </div>

      <div className="art-workspace">
        <div className="toolbar">
          <div className="tool-section">
            <h3>Tools</h3>
            <div className="tool-buttons">
              <motion.button
                className={`tool-btn ${tool === 'brush' ? 'active' : ''}`}
                onClick={() => setTool('brush')}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                🖌️
              </motion.button>
              <motion.button
                className={`tool-btn ${tool === 'eraser' ? 'active' : ''}`}
                onClick={() => setTool('eraser')}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                ⌫
              </motion.button>
              <motion.button
                className={`tool-btn ${tool === 'sticker' ? 'active' : ''}`}
                onClick={() => setTool('sticker')}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                😊
              </motion.button>
            </div>
          </div>

          <div className="tool-section">
            <h3>Colors</h3>
            <div className="color-palette">
              {COLORS.map((c) => (
                <motion.div
                  key={c}
                  className={`color-option ${color === c ? 'active' : ''}`}
                  style={{ backgroundColor: c }}
                  onClick={() => setColor(c)}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.95 }}
                />
              ))}
            </div>
          </div>

          <div className="tool-section">
            <h3>Brush Size</h3>
            <div className="brush-sizes">
              {BRUSH_SIZES.map((size) => (
                <motion.div
                  key={size}
                  className={`brush-size ${brushSize === size ? 'active' : ''}`}
                  onClick={() => setBrushSize(size)}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <div style={{ width: size, height: size }} />
                </motion.div>
              ))}
            </div>
          </div>

          {tool === 'sticker' && (
            <div className="tool-section">
              <h3>Stickers</h3>
              <div className="sticker-palette">
                {STICKERS.map((sticker) => (
                  <motion.button
                    key={sticker}
                    className={`sticker-option ${selectedSticker === sticker ? 'active' : ''}`}
                    onClick={() => setSelectedSticker(sticker)}
                    whileHover={{ scale: 1.2 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    {sticker}
                  </motion.button>
                ))}
              </div>
            </div>
          )}

          <div className="action-buttons">
            <motion.button
              className="action-btn clear"
              onClick={clearCanvas}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Clear All
            </motion.button>
            <motion.button
              className="action-btn save"
              onClick={saveDrawing}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Save Drawing
            </motion.button>
          </div>
        </div>

        <canvas
          ref={canvasRef}
          className="drawing-canvas"
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          onTouchStart={startDrawing}
          onTouchMove={draw}
          onTouchEnd={stopDrawing}
        />
      </div>
    </div>
  );
};

export default ArtStudio; 