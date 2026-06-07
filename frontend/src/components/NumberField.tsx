interface NumberFieldProps {
  id: string;
  label: string;
  value: number;
  hint: string;
  min?: number;
  max?: number;
  step?: number;
  onChange: (value: number) => void;
}

export function NumberField({
  id,
  label,
  value,
  hint,
  min = 0,
  max,
  step = 1,
  onChange,
}: NumberFieldProps) {
  return (
    <label className="field" htmlFor={id}>
      <span className="field__label">{label}</span>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-describedby={`${id}-hint`}
        required
      />
      <span className="field__hint" id={`${id}-hint`}>
        {hint}
      </span>
    </label>
  );
}
