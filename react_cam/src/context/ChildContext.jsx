import React, { createContext, useContext, useState } from 'react';

// Create the context for child data
const ChildContext = createContext();

export const ChildProvider = ({ children }) => {
  const [childData, setChildData] = useState(null); // Default null state

  return (
    <ChildContext.Provider value={{ childData, setChildData }}>
      {children}
    </ChildContext.Provider>
  );
};

// Custom hook to use the context
export const useChildContext = () => {
  const context = useContext(ChildContext);
  if (!context) {
    throw new Error('useChildContext must be used within a ChildProvider');
  }
  return context;
};
