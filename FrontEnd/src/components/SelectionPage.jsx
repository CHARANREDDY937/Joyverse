// export default SelectionPage; 
import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './SelectionPage.css';

const SelectionPage = () => {
  const navigate = useNavigate();

  const handleSelection = (role) => {
    navigate(`/${role}`); // Added backticks for template literal
  };

  const handleAdminClick = () => {
    navigate('/admin/login');
  };

  return (
    <div className="selection-container">
      <motion.button
        className="admin-button"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        whileHover={{ scale: 1.05 }}
        onClick={handleAdminClick}
      >
        SuperAdmin
      </motion.button>

      <motion.h1 
        className="selection-title"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
      >
        Pick Your Side!
      </motion.h1>

      <div className="cards-container">
        <motion.div
          className="selection-card child-card"
          initial={{ opacity: 0, x: -100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          whileHover={{ scale: 1.05 }}
          onClick={() => handleSelection('child')}
        >
          <div className="card-content">
            <motion.span 
              className="card-emoji"
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              👶
            </motion.span>
            <h2>I'm a Kid!</h2>
            <p>Let's Play & Learn!</p>
            <div className="card-features">
              <motion.div whileHover={{ scale: 1.1 }}>🎮 Play</motion.div>
              <motion.div whileHover={{ scale: 1.1 }}>🌟 Win</motion.div>
              <motion.div whileHover={{ scale: 1.1 }}>🎨 Create</motion.div>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="selection-card therapist-card"
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8 }}
          whileHover={{ scale: 1.05 }}
          onClick={() => handleSelection('therapist')}
        >
          <div className="card-content">
            <motion.span 
              className="card-emoji"
              animate={{ rotate: [0, 10, -10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              👩‍⚕️
            </motion.span>
            <h2>I'm a Therapist!</h2>
            <p>Guide & Support</p>
            <div className="card-features">
              <motion.div whileHover={{ scale: 1.1 }}>📊 Guide</motion.div>
              <motion.div whileHover={{ scale: 1.1 }}>📝 Plan</motion.div>
              <motion.div whileHover={{ scale: 1.1 }}>🤝 Help</motion.div>
            </div>
          </div>
        </motion.div>
      </div>

      <motion.div 
        className="background-decoration"
        animate={{ 
          scale: [1, 1.2, 1],
          rotate: [0, 360],
        }}
        transition={{ 
          duration: 20,
          repeat: Infinity,
          ease: "linear"
        }}
      >
        <div className="circle"></div>
        <div className="circle"></div>
        <div className="circle"></div>
      </motion.div>
    </div>
  );
};

export default SelectionPage;