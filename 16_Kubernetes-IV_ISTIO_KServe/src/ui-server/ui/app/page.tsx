import SimpleQuery from '../components/SimpleQuery'; // Importing the component

export default function Home() {
  return (
    <main className="container mx-auto p-4">
      <header className="text-center my-4">
        <h1 className="text-2xl font-bold">TSAI - EMLO V4</h1>
        <p className="text-gray-600">TSAI EMLO-4.0 Stable Diffusion model: Image Generation</p>
      </header>
      <SimpleQuery />  {/* Using the SimpleQuery component */}
    </main>
  );
}
