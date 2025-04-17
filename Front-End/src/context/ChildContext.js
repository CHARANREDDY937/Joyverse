   // joyverse/src/context/ChildContext.js
   import React, { createContext, useContext, useState } from 'react';

   const ChildContext = createContext();

   export const ChildProvider = ({ children }) => {
     const [childData, setChildData] = useState(null); // State to hold child's data

     return (
       <ChildContext.Provider value={{ childData, setChildData }}>
         {children}
       </ChildContext.Provider>
     );
   };

   export const useChildContext = () => {
     return useContext(ChildContext);
   };