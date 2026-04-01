"use client";

import { useState } from "react";

export type TabItem = {
  id: string;
  label: string;
  content: React.ReactNode;
};

export function Tabs({ items }: { items: TabItem[] }) {
  const [active, setActive] = useState(items[0]?.id ?? "");

  return (
    <div>
      <div className="flex flex-wrap gap-2 border-b border-slate/10 pb-3">
        {items.map((item) => {
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => setActive(item.id)}
              className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                isActive
                  ? "bg-ink text-white"
                  : "bg-white text-slate hover:text-ink"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>
      <div className="mt-6">
        {items.map((item) =>
          item.id === active ? <div key={item.id}>{item.content}</div> : null
        )}
      </div>
    </div>
  );
}
