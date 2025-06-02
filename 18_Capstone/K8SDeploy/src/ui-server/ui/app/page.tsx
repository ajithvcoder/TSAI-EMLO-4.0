import SimpleFileUploadApp1 from '../components/SimpleFileUploadApp1';
import SimpleFileUploadApp2 from '../components/SimpleFileUploadApp2';

export default function Home() {
  return (
    <main className="container mx-auto p-6 bg-gradient-to-br from-blue-200 via-purple-200 to-pink-200 min-h-screen">
      {/* Main Header */}
      <header className="text-center my-6">
        <h1 className="text-3xl font-extrabold text-purple-800 drop-shadow-md">
          TSAI - EMLO V4 - Capstone Project
        </h1>
        <p className="text-gray-700 text-lg font-medium">
          🌿 Vegetable Classifier & ⚽ Sports Classifier
        </p>
      </header>

      {/* Vegetable Classifier Section */}
      <section className="bg-white shadow-lg rounded-2xl p-6 mb-6 mx-auto w-3/4">
        <header className="flex justify-start pl-10">
          <h2 className="text-2xl font-bold text-green-600">🥦 Vegetable Classifier</h2>
        </header>
        <SimpleFileUploadApp1 />
      </section>

      {/* Sports Classifier Section */}
      <section className="bg-white shadow-lg rounded-2xl p-6 mx-auto w-3/4">
        <header className="flex justify-start pl-10">
          <h2 className="text-2xl font-bold text-blue-600">🏀 Sports Classifier</h2>
        </header>
        <SimpleFileUploadApp2 />
      </section>
    </main>
  );
}
