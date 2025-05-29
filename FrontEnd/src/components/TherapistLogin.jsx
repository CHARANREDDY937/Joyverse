// import React, { useState } from 'react';
// import { motion } from 'framer-motion';
// import { useNavigate } from 'react-router-dom';
// import { useChildContext } from '../context/ChildContext';
// import './TherapistLogin.css';

// const TherapistLogin = () => {
//   const navigate = useNavigate();
//   const { setUser } = useChildContext();
//   const [isLogin, setIsLogin] = useState(true);
//   const [formData, setFormData] = useState({
//     username: '',
//     password: '',
//     confirmPassword: '',
//     name: '',
//     licenseNumber: ''
//   });
//   const [error, setError] = useState(null);

//   const isProduction = process.env.NODE_ENV === 'production';
//   const NODE_BASE_URL = isProduction
//     ? 'https://backend-brmn.onrender.com'
//     : 'http://localhost:5000';

//   const validateUsername = (username) => {
//     const regex = /^[a-zA-Z0-9]{3,20}$/;
//     return regex.test(username);
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     setError(null);

//     if (!validateUsername(formData.username)) {
//       return setError('Username must be 3-20 alphanumeric characters');
//     }

//     try {
//       const url = isLogin
//         ? `${NODE_BASE_URL}/api/login`
//         : `${NODE_BASE_URL}/api/signup`;

//       const payload = {
//         username: formData.username,
//         password: formData.password,
//       };

//       if (!isLogin) {
//         if (formData.password !== formData.confirmPassword) {
//           return setError('Passwords do not match');
//         }
//         if (!formData.name) {
//           return setError('Full name is required for signup');
//         }
//         payload.confirmPassword = formData.confirmPassword;
//         payload.name = formData.name;
//       }

//       console.log(`Sending request to ${url} with payload:`, payload);

//       const response = await fetch(url, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify(payload),
//       });

//       if (!response.ok) {
//         const errorData = await response.json().catch(() => ({}));
//         console.error(`HTTP error ${response.status}:`, errorData);
//         if (response.status === 400) {
//           throw new Error(errorData.message || 'Invalid input. Please check your username and password.');
//         }
//         if (response.status === 403) {
//           throw new Error(errorData.message || 'Access denied. Please contact support.');
//         }
//         if (response.status === 404) {
//           throw new Error('Server endpoint not found. Please ensure the server is running.');
//         }
//         throw new Error(errorData.message || `Authentication failed (HTTP ${response.status})`);
//       }

//       const data = await response.json();

//       console.log('Response data:', data);

//       setUser({
//         username: formData.username,
//         role: 'therapist',
//         isAuthenticated: true
//       });

//       localStorage.setItem('therapistUsername', formData.username);
//       localStorage.setItem('userRole', 'therapist');

//       console.log('Navigating to /childlist');
//       navigate('/childlist', { replace: true });
//     } catch (error) {
//       console.error('Error during API call:', error.message);
//       setError(error.message || 'Failed to connect to server. Please check your network.');
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
//     <div className="therapist-login-container" data-state={isLogin ? "login" : "signup"}>
//       <motion.div
//         className="therapist-login-box"
//         initial={{ opacity: 0, y: 20 }}
//         animate={{ opacity: 1, y: 0 }}
//         transition={{ duration: 0.5 }}
//       >
//         <motion.div
//           className="therapist-header"
//           initial={{ scale: 0.8 }}
//           animate={{ scale: 1 }}
//           transition={{ duration: 0.5 }}
//         >
//           <span className="welcome-emoji">🧠</span>
//           <h1>{isLogin ? 'Welcome Back!' : 'Join as a Therapist!'}</h1>
//         </motion.div>

//         <form onSubmit={handleSubmit} className="therapist-form">
//           <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}>
//             <label htmlFor="username">Username</label>
//             <input
//               type="text"
//               id="username"
//               name="username"
//               value={formData.username}
//               onChange={handleChange}
//               placeholder="Enter username (3-20 alphanumeric)"
//               required
//             />
//           </motion.div>

//           <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
//             <label htmlFor="password">Password</label>
//             <input
//               type="password"
//               id="password"
//               name="password"
//               value={formData.password}
//               onChange={handleChange}
//               placeholder="Enter your password"
//               required
//             />
//           </motion.div>

//           {!isLogin && (
//             <>
//               <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}>
//                 <label htmlFor="confirmPassword">Confirm Password</label>
//                 <input
//                   type="password"
//                   id="confirmPassword"
//                   name="confirmPassword"
//                   value={formData.confirmPassword}
//                   onChange={handleChange}
//                   placeholder="Confirm your password"
//                   required
//                 />
//               </motion.div>

//               <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}>
//                 <label htmlFor="name">Therapist's Name</label>
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

//               <motion.div className="form-group" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 }}>
//                 <label htmlFor="licenseNumber">Professional License Number (Optional)</label>
//                 <input
//                   type="text"
//                   id="licenseNumber"
//                   name="licenseNumber"
//                   value={formData.licenseNumber}
//                   onChange={handleChange}
//                   placeholder="Enter license number"
//                 />
//               </motion.div>
//             </>
//           )}

//           <motion.button type="submit" className="submit-button" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
//             {isLogin ? 'Log In' : 'Sign Up'}
//           </motion.button>

//           {error && (
//             <motion.div className="error-message" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
//               <p>{error}</p>
//             </motion.div>
//           )}
//         </form>

//         <motion.div className="toggle-form" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}>
//           <p>
//             {isLogin ? "Need an account?" : "Already have an account?"}
//             <button className="toggle-button" onClick={handleToggle}>
//               {isLogin ? 'Sign Up' : 'Log In'}
//             </button>
//           </p>
//         </motion.div>
//       </motion.div>
//     </div>
//   );
// };

// export default TherapistLogin;

// TherapistAuth.jsx
import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChildContext } from '../context/ChildContext';
import './TherapistLogin.css'; // Assuming you named your CSS file this

const TherapistLogin = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [registrationSuccess, setRegistrationSuccess] = useState(false);
  const navigate = useNavigate();
  const { setUser } = useChildContext();
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    fullName: '',
    email: '',
    degree: '',
    specialization: '',
    certificate: null,
  });

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: files ? files[0] : value,
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const formPayload = new FormData();
    for (let key in formData) {
      if (formData[key]) {
        formPayload.append(key, formData[key]);
      }
    }

    fetch(`http://localhost:5000/therapist/${isLogin ? 'login' : 'register'}`, {
      method: 'POST',
      body: formPayload
    })
      .then(async (res) => {
        const contentType = res.headers.get("content-type");
        if (contentType && contentType.indexOf("application/json") !== -1) {
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.message || `HTTP error! status: ${res.status}`);
          }
          return data;
        } else {
          throw new Error('Server sent an invalid response');
        }
      })
      .then(data => {
        if (isLogin) {
          if (data.isVerified) {
            localStorage.setItem('therapistToken', data.token);
            setUser({
              username: formData.username,
              role: 'therapist',
              isAuthenticated: true
            });
            navigate('/childlist');
          } else {
            throw new Error('Your account is pending verification. Please wait for admin approval.');
          }
        } else {
          setRegistrationSuccess(true);
          setFormData({
            username: '',
            password: '',
            fullName: '',
            email: '',
            degree: '',
            specialization: '',
            certificate: null,
          });
        }
      })
      .catch(error => {
        console.error('Error:', error);
        alert(error.message || 'An error occurred');
      });
  };

  if (registrationSuccess) {
    return (
      <div className="therapist-login-container" data-state="success">
        <div className="therapist-login-box">
          <div className="therapist-header">
            <span className="header-icon">✅</span>
            <h1>Registration Successful!</h1>
            <div className="verification-message">
              <p>Thank you for registering as a therapist!</p>
              <p>Your application is currently under review.</p>
              <p>Please wait for admin verification before attempting to log in.</p>
              <p>We will process your application as soon as possible.</p>
            </div>
            <button 
              className="submit-button" 
              onClick={() => {
                setRegistrationSuccess(false);
                setIsLogin(true);
              }}
            >
              Return to Login
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="therapist-login-container" data-state={isLogin ? 'login' : 'signup'}>
      <div className="therapist-login-box">
        <div className="therapist-header">
          <span className="header-icon">🧠</span>
          <h1>{isLogin ? 'Therapist Login' : 'Therapist Sign Up'}</h1>
          <p className="subtitle">
            {isLogin ? 'Welcome back!' : 'Join the JoyVerse therapy team'}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="therapist-form">
          {!isLogin && (
            <>
              <div className="form-group">
                <label htmlFor="fullName">Full Name</label>
                <input type="text" id="fullName" name="fullName" required onChange={handleChange} />
              </div>

              <div className="form-group">
                <label htmlFor="email">Email</label>
                <input type="email" id="email" name="email" required onChange={handleChange} />
              </div>

              <div className="form-group">
                <label htmlFor="degree">Degree</label>
                <input type="text" id="degree" name="degree" required onChange={handleChange} />
              </div>

              <div className="form-group">
                <label htmlFor="specialization">Specialization</label>
                <input type="text" id="specialization" name="specialization" required onChange={handleChange} />
              </div>

              <div className="form-group">
                <label htmlFor="certificate">Upload Certificate</label>
                <input type="file" id="certificate" name="certificate" accept=".pdf,.jpg,.png" required onChange={handleChange} />
              </div>
            </>
          )}

          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input type="text" id="username" name="username" required onChange={handleChange} />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input type="password" id="password" name="password" required onChange={handleChange} />
          </div>

          <button type="submit" className="submit-button">
            {isLogin ? 'Login' : 'Register'}
          </button>
        </form>

        <div className="toggle-form">
          {isLogin ? "Don't have an account?" : 'Already have an account?'}
          <button className="toggle-button" onClick={() => setIsLogin(!isLogin)}>
            {isLogin ? 'Sign up' : 'Login'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TherapistLogin;