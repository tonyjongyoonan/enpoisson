import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import './Register.css';

function Register() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');

  const handleRegister = async (event) => {
    event.preventDefault();
    // Handle the registration logic here
    console.log('Registering:', username, email, password);
    const response = await fetch('http://localhost:8000/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, email, password })
    });

    const data = await response.json();
    console.log('Response: ', data);
  };

  return (
    <div className="register-container">
      <form className="register-form" onSubmit={handleRegister}>
        <h2 className="register-heading">Register</h2>
        <div className="register-input-container">
          <input
            className="register-input"
            type="text"
            placeholder="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div className="register-input-container">
          <input
            className="register-input"
            type="email"
            placeholder="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="register-input-container">
          <input
            className="register-input"
            type="password"
            placeholder="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <div className="register-button-container">
          <button className="register-button" type="submit">Register</button>
        </div>
        <p className="register-login-link">
          Already have an account? <br /> <Link to="/login" className="login-link"> Log In</Link>        </p>
      </form>
    </div>
  );
}

export default Register;
