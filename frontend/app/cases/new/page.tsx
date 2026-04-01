import Link from "next/link";

export default function NewCasePage() {
  return (
    <div className="px-6 py-10 lg:px-12">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-slate">
            New case
          </p>
          <h1 className="mt-2 font-[var(--font-fraunces)] text-3xl text-ink">
            Upload a study
          </h1>
        </div>
        <Link
          href="/dashboard"
          className="rounded-full border border-slate/20 px-4 py-2 text-xs font-semibold text-slate"
        >
          Back to dashboard
        </Link>
      </header>

      <form
        className="mt-8 grid gap-6 rounded-3xl border border-slate/10 bg-white/80 p-8 shadow-soft"
        action="/api/cases"
        method="post"
        encType="multipart/form-data"
      >
        <div className="grid gap-4 md:grid-cols-2">
          <label className="grid gap-2 text-sm font-medium text-slate">
            Patient name
            <input
              type="text"
              name="patientName"
              placeholder="Dr. Smith, Jane Doe"
              className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
            />
          </label>
          <label className="grid gap-2 text-sm font-medium text-slate">
            Patient ID
            <input
              type="text"
              name="patientId"
              placeholder="CT-1044"
              className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
            />
          </label>
        </div>

        <div className="rounded-2xl border border-slate/10 bg-slate/5 p-4">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.1em] text-slate">
            Clinical Data
          </p>
          <div className="grid gap-4 md:grid-cols-3">
            <label className="grid gap-2 text-sm font-medium text-slate">
              Age
              <input
                type="number"
                name="age"
                placeholder="65"
                min="0"
                max="120"
                className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
              />
            </label>
            <label className="grid gap-2 text-sm font-medium text-slate">
              Smoking Status
              <select
                name="isSmoker"
                className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
              >
                <option value="">Unknown</option>
                <option value="true">Smoker (Current/Former)</option>
                <option value="false">Never Smoker</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-medium text-slate">
              Pack Years
              <input
                type="number"
                name="packYears"
                placeholder="30"
                min="0"
                className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
              />
            </label>
            <label className="grid gap-2 text-sm font-medium text-slate">
              ECOG Performance
              <select
                name="ecog"
                className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
              >
                <option value="">Unknown</option>
                <option value="0">0 - Fully active</option>
                <option value="1">1 - Restricted activity</option>
                <option value="2">2 - Ambulatory, can't work</option>
                <option value="3">3 - Limited self-care</option>
                <option value="4">4 - Bedridden</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-medium text-slate md:col-span-2">
              Histology
              <select
                name="histology"
                className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
              >
                <option value="">Unknown</option>
                <option value="Adenocarcinoma">Adenocarcinoma</option>
                <option value="Squamous cell carcinoma">Squamous cell carcinoma</option>
                <option value="Large cell carcinoma">Large cell carcinoma</option>
                <option value="Small cell carcinoma">Small cell carcinoma</option>
                <option value="Other">Other</option>
              </select>
            </label>
          </div>
        </div>

        <label className="grid gap-2 text-sm font-medium text-slate">
          Modality
          <select
            name="modality"
            className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
            defaultValue="CT"
          >
            <option value="CT">CT</option>
            <option value="X-ray">X-ray</option>
            <option value="Pathology">Pathology</option>
          </select>
        </label>

        <label className="grid gap-2 text-sm font-medium text-slate">
          Study file
          <input
            type="file"
            name="file"
            required
            className="rounded-xl border border-dashed border-slate/30 bg-mist px-4 py-6 text-sm"
          />
        </label>

        <label className="grid gap-2 text-sm font-medium text-slate">
          Notes
          <textarea
            name="notes"
            rows={4}
            placeholder="Optional clinical context or referral note."
            className="rounded-xl border border-slate/20 px-4 py-2 text-sm text-ink"
          />
        </label>

        <div className="flex flex-wrap items-center gap-4">
          <button
            type="submit"
            className="rounded-full bg-ink px-6 py-2 text-sm font-semibold text-white"
          >
            Create case
          </button>
          <p className="text-xs text-slate">
            Uploading triggers storage only. Run inference from the case view.
          </p>
        </div>
      </form>
    </div>
  );
}
