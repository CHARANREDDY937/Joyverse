import React, { createContext, useContext, useState } from 'react';

export const ChildContext = createContext();

export const ChildProvider = ({ children }) => {
  const [user, setUser] = useState({
    username: null,
    role: null,
    isAuthenticated: false
  });
  const [childData, setChildData] = useState({
    username: null,
    name: null,
    progressData: null
  });

  return (
    <ChildContext.Provider value={{ user, setUser, childData, setChildData }}>
      {children}
    </ChildContext.Provider>
  );
};

export const useChildContext = () => useContext(ChildContext);