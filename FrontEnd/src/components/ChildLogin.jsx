// import React, { useState } from 'react';
// import { motion } from 'framer-motion';
// import { useNavigate } from 'react-router-dom';
// import './ChildLogin.css';

// const ChildLogin = () => {
//   const navigate = useNavigate();
//   const [isLogin, setIsLogin] = useState(true);
//   const [formData, setFormData] = useState({
//     username: '',
//     password: '',
//     confirmPassword: '',
//     name: ''
//   });
//   const [error, setError] = useState(null);

//   // const handleSubmit = async (e) => {
//   //   e.preventDefault();
//   //   setError(null);

//   //   try {
//   //     const url = isLogin
//   //       ? 'http://localhost:5000/api/login'
//   //       : 'http://localhost:5000/api/signup';

//   //     const payload = {
//   //       username: formData.username,
//   //       password: formData.password,
//   //     };

//   //     // Add confirmPassword and name if it's a signup request
//   //     if (!isLogin) {
//   //       if (formData.password !== formData.confirmPassword) {
//   //         return setError('Passwords do not match');
//   //       }
//   //       payload.confirmPassword = formData.confirmPassword;
//   //       payload.name = formData.name;
//   //     }

//   //     const response = await fetch(url, {
//   //       method: 'POST',
//   //       headers: { 'Content-Type': 'application/json' },
//   //       body: JSON.stringify(payload),
//   //     });

//   //     const data = await response.json();

//   //     if (response.ok) {
//   //       if (isLogin) {
//   //         // Log the login event to MongoDB
//   //         await fetch('http://localhost:5000/api/login', {
//   //           method: 'POST',
//   //           headers: { 'Content-Type': 'application/json' },
//   //           body: JSON.stringify({ username: formData.username }),
//   //         });
//   //       }
//   //       navigate('/child/games'); // Login or signup success
//   //     } else {
//   //       setError(data.message || 'An error occurred');
//   //     }
//   //   } catch (error) {
//   //     console.error('Error during API call:', error);
//   //     setError('Failed to connect to server');
//   //   }
//   // };
//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setError(null);
  
//     try {
//       const url = isLogin
//         ? 'http://localhost:5000/api/login'
//         : 'http://localhost:5000/api/signup';
  
//       const payload = {
//         username: formData.username,
//         password: formData.password,
//       };
  
//       if (!isLogin) {
//         if (formData.password !== formData.confirmPassword) {
//           return setError('Passwords do not match');
//         }
//         payload.confirmPassword = formData.confirmPassword;
//         payload.name = formData.name;
//       }
  
//       const response = await fetch(url, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(payload),
//       });
  
//       const data = await response.json();
  
//       if (response.ok) {
//         // ✅ Store username in localStorage
//         localStorage.setItem('childUserId', formData.username);
  
//         if (isLogin) {
//           // Optional: Log login event again (this is already done above)
//           await fetch('http://localhost:5000/api/login', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ username: formData.username }),
//           });
//         }
  
//         // ✅ Navigate to game page
//         navigate('/child/games');
//       } else {
//         setError(data.message || 'An error occurred');
//       }
//     } catch (error) {
//       console.error('Error during API call:', error);
//       setError('Failed to connect to server');
//     }
//   };
  
  

//   const handleChange = (e) => {
//     setFormData({
//       ...formData,
//       [e.target.name]: e.target.value,
//     });
//   };

//   const handleToggle = () => {
//     setIsLogin(!isLogin);
//     setError(null);
//   };

//   return (
//     <div className="child-login-container" data-state={isLogin ? "login" : "signup"}>
//       <motion.div
//         className="login-box"
//         initial={{ opacity: 0, y: 20 }}
//         animate={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.5 }}
//       >
//         <motion.div
//           className="login-header"
//           initial={{ scale: 0.8 }}
//           animate={{ scale: 1 }}
//           transition={{ duration: 0.5 }}
//         >
//           <span className="welcome-emoji">👋</span>
//           <h1>{isLogin ? 'Welcome Back!' : 'Join the Fun!'}</h1>
//         </motion.div>

//         <form onSubmit={handleSubmit} className="login-form">
//           <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
//             <label htmlFor="username">Your Name</label>
//             <input
//               type="text"
//               id="username"
//               name="username"
//               value={formData.username}
//               onChange={handleChange}
//               placeholder="Enter your name"
//               required
//             />
//           </motion.div>

//           <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
//             <label htmlFor="password">Secret Code</label>
//             <input
//               type="password"
//               id="password"
//               name="password"
//               value={formData.password}
//               onChange={handleChange}
//               placeholder="Enter your secret code"
//               required
//             />
//           </motion.div>

//           {!isLogin && (
//             <>
//               <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
//                 <label htmlFor="confirmPassword">Confirm Secret Code</label>
//                 <input
//                   type="password"
//                   id="confirmPassword"
//                   name="confirmPassword"
//                   value={formData.confirmPassword}
//                   onChange={handleChange}
//                   placeholder="Enter your secret code again"
//                   required
//                 />
//               </motion.div>

//               <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}>
//                 <label htmlFor="name">Child's Full Name</label>
//                 <input
//                   type="text"
//                   id="name"
//                   name="name"
//                   value={formData.name}
//                   onChange={handleChange}
//                   placeholder="Enter your full name"
//                   required
//                 />
//               </motion.div>
//             </>
//           )}

//           <motion.button type="submit" className="submit-button" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
//             {isLogin ? 'Let\'s Play!' : 'Join Now!'}
//           </motion.button>

//           {error && (
//             <motion.div className="error-message" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
//               <p>{error}</p>
//             </motion.div>
//           )}
//         </form>

//         <motion.div className="toggle-form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
//           <p>
//             {isLogin ? "Don't have an account?" : "Already have an account?"}
//             <button className="toggle-button" onClick={handleToggle}>
//               {isLogin ? 'Join Now!' : 'Sign In!'}
//             </button>
//           </p>
//         </motion.div>
//       </motion.div>
//     </div>
//   );
// };
// export default ChildLogin;

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
        payload.confirmPassword = formData.confirmPassword;
        payload.name = formData.name;
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        // Store username in localStorage
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
            </>
          )}

          <motion.button type="submit" className="submit-button" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            {isLogin ? 'Let\'s Play!' : 'Join Now!'}
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