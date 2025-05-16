import React, { createContext, useContext, useState } from 'react';

const ChildContext = createContext();

export const ChildProvider = ({ children }) => {
  const [childData, setChildData] = useState({
    username: null,
    name: null,
    progressData: null
  });

  return (
    <ChildContext.Provider value={{ childData, setChildData }}>
      {children}
    </ChildContext.Provider>
  );
};

export const useChildContext = () => useContext(ChildContext);
