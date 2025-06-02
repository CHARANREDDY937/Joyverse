import React, { useState, useContext } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ChildContext } from '../context/ChildContext';
import './ChildLogin.css';

const ChildLogin = () => {
  const navigate = useNavigate();
  const { setUser } = useContext(ChildContext);
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    confirmPassword: '',
    name: '',
    age: ''
  });
  const [error, setError] = useState(null);

  const isProduction = process.env.NODE_ENV === 'production';
  const NODE_BASE_URL = isProduction
    ? 'https://backend-brmn.onrender.com'
    : 'http://localhost:5000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const url = isLogin
        ? `${NODE_BASE_URL}/api/login`
        : `${NODE_BASE_URL}/api/signup`;

      const payload = {
        username: formData.username,
        password: formData.password,
      };

      if (!isLogin) {
        if (formData.password !== formData.confirmPassword) {
          return setError('Passwords do not match');
        }

        const age = parseInt(formData.age);
        if (isNaN(age) || age > 12) {
          return setError('Sorry, this app is only for kids aged 12 or below.');
        }

        payload.confirmPassword = formData.confirmPassword;
        payload.name = formData.name;
        payload.age = formData.age;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        setUser({
          username: formData.username,
          role: 'child',
          isAuthenticated: true
        });
        localStorage.setItem('childUserId', formData.username);
        navigate('/child/games');
      } else {
        setError(data.message || 'An error occurred');
      }
    } catch (error) {
      console.error('Error during API call:', error);
      setError('Failed to connect to server');
    }
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleToggle = () => {
    setIsLogin(!isLogin);
    setError(null);
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
          <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
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

          <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
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
            <>
              <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
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

              <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}>
                <label htmlFor="name">Child's Full Name</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  placeholder="Enter your full name"
                  required
                />
              </motion.div>

              <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 }}>
                <label htmlFor="age">Your Age</label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="Enter your age"
                  required
                  min="1"
                />
              </motion.div>
            </>
          )}

          <motion.button type="submit" className="submit-button" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            {isLogin ? "Let's Play!" : "Join Now!"}
          </motion.button>

          {error && (
            <motion.div className="error-message" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
              <p>{error}</p>
            </motion.div>
          )}
        </form>

        <motion.div className="toggle-form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
          <p>
            {isLogin ? "Don't have an account?" : "Already have an account?"}
            <button className="toggle-button" onClick={handleToggle}>
              {isLogin ? 'Join Now!' : 'Sign In!'}
            </button>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ChildLogin;
