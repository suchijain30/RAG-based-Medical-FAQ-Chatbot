import React, { useEffect, useState } from 'react';
import { onAuthStateChanged, setPersistence, browserLocalPersistence, signOut } from 'firebase/auth';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatView from './components/ChatView';
import DoctorSearchView from './components/DoctorSearchView';
import SymptomTrackerView from './components/SymptomTrackerView';
import PatientGuidesView from './components/PatientGuidesView';
import OnboardingModal from './components/OnboardingModal';
import AuthScreen from './components/AuthScreen';
import { auth, firebaseConfigError } from './services/firebase';

function getUserProfile(uid) {
  try {
    return JSON.parse(localStorage.getItem(`medibot_profile_${uid}`) || '{}');
  } catch (error) {
    console.error('Unable to load the saved profile:', error);
    return {};
  }
}

function saveUserProfile(uid, profile) {
  localStorage.setItem(`medibot_profile_${uid}`, JSON.stringify(profile));
}

export default function App() {
  const [currentTab, setCurrentTab] = useState('chat'); // 'chat' | 'symptoms' | 'doctors' | 'guides'
  const [user, setUser] = useState(null);
  const [isInitializing, setIsInitializing] = useState(Boolean(auth));
  const [authError, setAuthError] = useState(
    auth ? '' : `Firebase configuration is missing: ${firebaseConfigError.join(', ')}.`,
  );
  const [city, setCity] = useState('');
  const [showProfileModal, setShowProfileModal] = useState(false);

  useEffect(() => {
    if (!auth) {
      return undefined;
    }

    let active = true;
    let unsubscribe;
    setPersistence(auth, browserLocalPersistence)
      .then(() => {
        if (!active) return;
        unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
          if (!active) return;
          if (!firebaseUser) {
            setUser(null);
            setCity('');
            setAuthError('');
            setIsInitializing(false);
            return;
          }

          const profile = getUserProfile(firebaseUser.uid);
          const nextUser = {
            ...profile,
            uid: firebaseUser.uid,
            email: firebaseUser.email,
            fullName: profile.fullName || firebaseUser.displayName || firebaseUser.email,
          };
          setUser(nextUser);
          setCity(nextUser.city || 'New York, NY');
          setAuthError('');
          setIsInitializing(false);
        }, (error) => {
          setAuthError(`Unable to restore your sign-in: ${error.message}`);
          setIsInitializing(false);
        });
      })
      .catch((error) => {
        setAuthError(`Unable to initialize Firebase Authentication: ${error.message}`);
        setIsInitializing(false);
      });

    return () => {
      active = false;
      unsubscribe?.();
    };
  }, []);

  const handleCityChange = (newCity) => {
    setCity(newCity);
    const updated = { ...user, city: newCity };
    setUser(updated);
    saveUserProfile(user.uid, updated);
  };

  const handleDiscussWithBot = () => {
    setCurrentTab('chat');
  };

  const handleRequestNewGuide = () => {
    setCurrentTab('chat');
  };

  const handleLogout = async () => {
    if (!window.confirm('Log out from MediBot?')) return;
    try {
      await signOut(auth);
      setAuthError('');
    } catch (error) {
      setAuthError(`Unable to log out: ${error.message}`);
    }
  };

  if (isInitializing) {
    return <main className="auth-page"><p className="auth-subtitle">Restoring your secure session…</p></main>;
  }

  if (!user) {
    return (
      <AuthScreen
        configError={!auth ? `Firebase configuration is missing: ${firebaseConfigError.join(', ')}.` : ''}
        initialError={authError}
      />
    );
  }

  return (
    <div className="app-container">
      {/* Left Sidebar */}
      <Sidebar 
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        onOpenProfile={() => setShowProfileModal(true)}
        onLogout={handleLogout}
      />

      {/* Main Wrapper */}
      <div className="main-wrapper">
        {authError && <p className="app-auth-error" role="alert">{authError}</p>}
        {/* Top Header */}
        <Header 
          city={city}
          onCityChange={handleCityChange}
          onOpenProfile={() => setShowProfileModal(true)}
          user={user}
        />

        {/* View Switcher */}
        {currentTab === 'chat' && (
          <ChatView 
            city={city}
            user={user}
            onFindSpecialist={() => setCurrentTab('doctors')}
          />
        )}

        {currentTab === 'doctors' && (
          <DoctorSearchView city={city} />
        )}

        {currentTab === 'symptoms' && (
          <SymptomTrackerView onDiscussWithBot={handleDiscussWithBot} />
        )}

        {currentTab === 'guides' && (
          <PatientGuidesView onRequestNewGuide={handleRequestNewGuide} />
        )}
      </div>

      {/* Onboarding / Profile Modal */}
      {showProfileModal && (
        <OnboardingModal 
          user={user}
          onComplete={(updated) => {
            setUser(updated);
            setCity(updated.city);
            saveUserProfile(user.uid, updated);
            setShowProfileModal(false);
          }}
          onClose={() => setShowProfileModal(false)}
        />
      )}
    </div>
  );
}
