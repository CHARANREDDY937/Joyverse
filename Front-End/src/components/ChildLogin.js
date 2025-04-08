import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './ChildLogin.css';

const ChildLogin = () => {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    confirmPassword: '',
    name: ''
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Add your authentication logic here
      console.log('Form submitted:', formData);
      
      // After successful authentication
      if (isLogin) {
        // Navigate to games dashboard after successful login
        navigate('/child/games');
      } else {
        // After successful registration, automatically log in and navigate
        navigate('/child/games');
      }
    } catch (error) {
      console.error('Authentication error:', error);
      // Handle error appropriately
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleToggle = () => {
    setIsLogin(!isLogin);
  };

  return (
    <div className="child-login-container" data-state={isLogin ? "login" : "signup"}>
      <motion.div 
        className="login-box"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div 
          className="login-header"
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <span className="welcome-emoji">👋</span>
          <h1>{isLogin ? 'Welcome Back!' : 'Join the Fun!'}</h1>
        </motion.div>

        <form onSubmit={handleSubmit} className="login-form">
          <motion.div 
            className="form-group"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <label htmlFor="username">Your Name</label>
            <input
              type="text"
              id="username"
              name="username"
              value={formData.username}
              onChange={handleChange}
              placeholder="Enter your name"
              required
            />
          </motion.div>

          <motion.div 
            className="form-group"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <label htmlFor="password">Secret Code</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Enter your secret code"
              required
            />
          </motion.div>

          {!isLogin && (
            <motion.div 
              className="form-group"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <label htmlFor="confirmPassword">Confirm Secret Code</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Enter your secret code again"
                required
              />
            </motion.div>
          )}

          <motion.button
            type="submit"
            className="submit-button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isLogin ? 'Let\'s Play!' : 'Join Now!'}
          </motion.button>
        </form>

        <motion.div 
          className="toggle-form"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <p>
            {isLogin ? "Don't have an account?" : "Already have an account?"}
            <button 
              className="toggle-button"
              onClick={handleToggle}
            >
              {isLogin ? 'Join Now!' : 'Sign In!'}
            </button>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ChildLogin; 