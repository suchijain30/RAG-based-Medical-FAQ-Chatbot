import React, { useEffect, useRef, useState } from 'react';
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
import { fetchProfile, saveProfile } from './services/profile';

export default function App() {
  const [currentTab, setCurrentTab] = useState('chat'); // 'chat' | 'symptoms' | 'doctors' | 'guides'
  const [user, setUser] = useState(null);
  const [isInitializing, setIsInitializing] = useState(Boolean(auth));
  const [authError, setAuthError] = useState(
    auth ? '' : `Firebase configuration is missing: ${firebaseConfigError.join(', ')}.`,
  );
  const [city, setCity] = useState('');
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('medibot_theme') || 'light');
  const signupProfiles = useRef({});

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem('medibot_theme', theme);
  }, [theme]);

  useEffect(() => {
    if (!auth) {
      return undefined;
    }

    let active = true;
    let unsubscribe;
    setPersistence(auth, browserLocalPersistence)
      .then(() => {
        if (!active) return;
        unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
          if (!active) return;
          if (!firebaseUser) {
            setUser(null);
            setCity('');
            setAuthError('');
            setIsInitializing(false);
            return;
          }

          try {
            const profile = {
              ...await fetchProfile(firebaseUser),
              ...signupProfiles.current[firebaseUser.uid],
            };
            if (!active) return;
            const nextUser = { ...profile, uid: firebaseUser.uid, email: firebaseUser.email };
            setUser(nextUser);
            setCity(nextUser.city || '');
            setAuthError('');
            setShowProfileModal(!profile.fullName || !profile.city || !profile.gender || !profile.dob);
            setIsInitializing(false);
          } catch (error) {
            if (!active) return;
            setUser({
              uid: firebaseUser.uid,
              email: firebaseUser.email,
              fullName: firebaseUser.displayName || '',
              city: '',
              gender: '',
              dob: '',
            });
            setCity('');
            setShowProfileModal(true);
            setAuthError(`Unable to load your profile: ${error.message}`);
            setIsInitializing(false);
          }
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

  const handleCityChange = async (newCity) => {
    const updated = { ...user, city: newCity };
    try {
      await saveProfile(updated, user);
      setUser(updated);
      setCity(newCity);
      setAuthError('');
    } catch (error) {
      setAuthError(`Unable to save your location: ${error.message}`);
    }
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
        onProfileSaved={(profile, firebaseUser) => {
          signupProfiles.current[firebaseUser.uid] = profile;
          setUser({ ...profile, uid: firebaseUser.uid, email: firebaseUser.email });
          setCity(profile.city);
          setShowProfileModal(false);
          setAuthError('');
          setIsInitializing(false);
        }}
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
          theme={theme}
          onToggleTheme={() => setTheme(theme === 'light' ? 'dark' : 'light')}
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
          <DoctorSearchView city={city} user={user} />
        )}

        {currentTab === 'symptoms' && (
          <SymptomTrackerView user={user} city={city} onDiscussWithBot={handleDiscussWithBot} />
        )}

        {currentTab === 'guides' && (
          <PatientGuidesView onRequestNewGuide={handleRequestNewGuide} />
        )}
      </div>

      {/* Onboarding / Profile Modal */}
      {showProfileModal && (
        <OnboardingModal 
          user={user}
          onComplete={async (updated) => {
            try {
              const saved = await saveProfile(updated, user);
              setUser({ ...user, ...saved });
              setCity(saved.city);
              setShowProfileModal(false);
              setAuthError('');
            } catch (error) {
              setAuthError(`Unable to save your profile: ${error.message}`);
              throw error;
            }
          }}
          onClose={() => setShowProfileModal(false)}
        />
      )}
    </div>
  );
}
