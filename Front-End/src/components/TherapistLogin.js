// joyverse/src/components/TherapistLogin.js
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useChildContext } from '../context/ChildContext'; 
import './TherapistLogin.css';

const TherapistLogin = () => {
  const navigate = useNavigate();
  const { setChildData } = useChildContext(); 
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    fullName: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // Set child's data in context
    setChildData({
      fullName: formData.fullName,
      email: formData.email,
    });
    // Navigate to the child's info page
    navigate('/child-info'); 
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
    <div className="therapist-login-container" data-state={isLogin ? "login" : "signup"}>
      <motion.div 
        className="therapist-login-box"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <motion.div 
          className="therapist-header"
          initial={{ scale: 0.8 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <div className="header-icon">🧠</div>
          <h1>{isLogin ? 'Welcome Back' : 'Join Our Network'}</h1>
          <p className="subtitle">{isLogin ? 'Continue your journey of helping others' : 'Start making a difference today'}</p>
        </motion.div>

        <form onSubmit={handleSubmit} className="therapist-form">
          {/* Full Name input for signup */}
          {!isLogin && (
            <motion.div 
              className="form-group"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <label htmlFor="fullName">Full Name</label>
              <input
                type="text"
                id="fullName"
                name="fullName"
                value={formData.fullName}
                onChange={handleChange}
                placeholder="Enter your full name"
                required
              />
            </motion.div>
          )}

          {/* Email input */}
          <motion.div 
            className="form-group"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <label htmlFor="email">Email Address</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleChange}
              placeholder="Enter your email"
              required
            />
          </motion.div>

          {/* Password input */}
          <motion.div 
            className="form-group"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              placeholder="Enter your password"
              required
            />
          </motion.div>

          {/* Confirm Password input for signup */}
          {!isLogin && (
            <motion.div 
              className="form-group"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.5 }}
            >
              <label htmlFor="confirmPassword">Confirm Password</label>
              <input
                type="password"
                id="confirmPassword"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm your password"
                required
              />
            </motion.div>
          )}

          {/* Submit button */}
          <motion.button
            type="submit"
            className="submit-button"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            {isLogin ? 'Sign In' : 'Create Account'}
          </motion.button>
        </form>

        {/* Toggle between login and signup */}
        <motion.div 
          className="toggle-form"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
        >
          <p>
            {isLogin ? "Don't have an account?" : "Already have an account?"}
            <button 
              className="toggle-button"
              onClick={handleToggle}
            >
              {isLogin ? 'Register' : 'Sign In'}
            </button>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default TherapistLogin;