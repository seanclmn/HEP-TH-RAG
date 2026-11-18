/** Join class names, skipping falsy ones: cn("btn", active && "btn-active"). */
export const cn = (...classes: (string | false | null | undefined)[]) =>
  classes.filter(Boolean).join(" ");
